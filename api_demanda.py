"""
Servicio FastAPI de prediccion de demanda - SOBUSA Data Challenge 2026.

Sirve los modelos que entrena modelo_demanda.py (carpeta modelos/) y calcula
las variables con las mismas funciones del entrenamiento, para que el modelo
reciba al predecir exactamente lo que vio al entrenar.

Al arrancar carga en memoria la copia local de datos/ que genera
extraer_datos.py (subidas por ruta y franja de 15 min de 2023 a 2025, ya
limpias con las mismas reglas del entrenamiento), el calendario y los nombres
de las rutas. Cada prediccion se resuelve en memoria, sin consultar la base.
Para tomar datos nuevos, corra extraer_datos.py y reinicie el servicio.

Ejecucion:
    uvicorn api_demanda:app --reload        (o: python api_demanda.py)
Documentacion interactiva: http://127.0.0.1:8000/docs
"""

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

import modelo_demanda as md

CARPETA_MODELOS = Path(__file__).resolve().parent / "modelos"
ZONA_HORARIA = "America/Bogota"     # las franjas de la base estan en hora local
HORAS_MINIMAS_HISTORIA = 48         # sin datos en las 48 h previas no se predice (cubre
                                    # el dia sin sensor de fin de mes)
DIAS_HISTORIA = 371                 # lag_364d mira 364 dias atras, con margen

estado: dict = {}


# ---------------------------------------------------------------------------
# CARGA AL ARRANCAR
# ---------------------------------------------------------------------------
def _cargar_modelos() -> dict:
    modelos = {}
    for h in md.HORIZONTES_MIN:
        base = CARPETA_MODELOS / f"demanda_h{h}"
        try:
            with open(f"{base}_meta.json", encoding="utf-8") as f:
                meta = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(
                f"No existe {base}_meta.json. Entrene primero: python modelo_demanda.py"
            ) from None
        # Un modelo entrenado con otras variables predeciria basura en silencio
        if meta["feature_version"] != md.FEATURE_VERSION or meta["features"] != md.FEATURES:
            raise RuntimeError(
                f"{base.name} se entreno con variables '{meta['feature_version']}' y el "
                f"codigo usa '{md.FEATURE_VERSION}'. Reentrene: python modelo_demanda.py"
            )
        if meta.get("fuente") != "sensores":
            print(f"AVISO: {base.name} se entreno con datos '{meta.get('fuente')}', no reales.")
        modelo = xgb.XGBRegressor()
        modelo.load_model(f"{base}.json")
        modelos[h] = (modelo, meta)
    return modelos


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not md.ARCHIVO_DEMANDA.exists():
        raise RuntimeError(f"Falta {md.ARCHIVO_DEMANDA}: corra primero python extraer_datos.py")
    estado["modelos"] = _cargar_modelos()
    estado["serie"] = md.preparar_serie(md.cargar_datos())
    estado["calendario"] = md.cargar_calendario()
    rutas = pd.read_parquet(md.CARPETA_DATOS / "rutas.parquet")
    estado["nombres"] = dict(zip(rutas["id_ruta"].astype(int), rutas["nombre"]))
    estado["cargado_en"] = datetime.now(timezone.utc)
    obs = estado["serie"][estado["serie"]["observado"]]
    print(f"Listo: {len(obs):,} franjas con servicio, hasta {obs['ts'].max()}")
    yield
    estado.clear()


app = FastAPI(
    title="SOBUSA - Prediccion de demanda",
    description="Subidas de pasajeros por ruta y franja de 15 min, a 30, 60 y 120 minutos (RF-005).",
    version=md.FEATURE_VERSION,
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# ESQUEMAS DE RESPUESTA
# ---------------------------------------------------------------------------
class Prediccion(BaseModel):
    horizonte_min: int
    franja_objetivo: datetime = Field(description="Inicio de la franja de 15 min que se predice")
    subidas_predichas: float
    linea_base_prom_4_semanas: float | None = Field(
        description="Promedio de la misma franja en las 4 semanas previas, como referencia"
    )
    subidas_reales: float | None = Field(
        description="Valor observado, solo si la franja ya esta en la base (sirve para comparar)"
    )
    servicio_habitual: bool = Field(
        description="La ruta opero en esta franja el mismo dia de la semana en alguna de las 4 "
        "semanas previas. Si es false, la prediccion es poco confiable: el modelo solo aprendio "
        "de franjas con servicio."
    )
    model_id: str


class RespuestaPrediccion(BaseModel):
    id_ruta: int
    nombre_ruta: str | None
    referencia: datetime = Field(description="Ultima franja de 15 min cuyo dato se da por conocido")
    feature_version: str
    predicciones: list[Prediccion]


class Ruta(BaseModel):
    id_ruta: int
    nombre: str | None
    primera_franja: datetime
    ultima_franja: datetime
    franjas_con_servicio: int


# ---------------------------------------------------------------------------
# PREDICCION
# ---------------------------------------------------------------------------
def _o_nada(valor) -> float | None:
    return None if pd.isna(valor) else round(float(valor), 2)


def _normalizar_referencia(referencia: datetime | None, obs_ruta: pd.DataFrame) -> pd.Timestamp:
    if referencia is None:
        return obs_ruta["ts"].max()
    ts = pd.Timestamp(referencia)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(ZONA_HORARIA).tz_localize(None)
    return ts.floor(f"{md.PASO_MIN}min")


def _serie_conocida(ruta: pd.DataFrame, referencia: pd.Timestamp, hasta: pd.Timestamp) -> pd.DataFrame:
    """Grilla regular de la ruta desde DIAS_HISTORIA dias antes de la referencia
    hasta la ultima franja objetivo. Lo posterior a la referencia se borra: al
    predecir solo se conoce lo ocurrido hasta la referencia."""
    inicio = max(referencia.normalize() - pd.Timedelta(days=DIAS_HISTORIA), ruta["ts"].min())
    grilla = pd.date_range(inicio, hasta, freq=f"{md.PASO_MIN}min")
    g = ruta.set_index("ts").reindex(grilla).rename_axis("ts").reset_index()
    g["id_ruta"] = ruta["id_ruta"].iloc[0]
    g.loc[g["ts"] > referencia, ["subidas", "bajadas", "nro_viajes"]] = np.nan
    g["observado"] = g["subidas"].notna()
    return g


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def inicio():
    return RedirectResponse("/docs")


@app.get("/salud")
def salud():
    obs = estado["serie"][estado["serie"]["observado"]]
    return {
        "estado": "ok",
        "horizontes_min": sorted(estado["modelos"]),
        "datos_cargados_en": estado["cargado_en"],
        "ultima_franja_con_datos": obs["ts"].max(),
    }


@app.get("/modelos")
def modelos():
    """Metadatos de cada modelo (seccion 9.2), con sus metricas de prueba."""
    return [estado["modelos"][h][1] for h in sorted(estado["modelos"])]


@app.get("/rutas", response_model=list[Ruta])
def rutas():
    obs = estado["serie"][estado["serie"]["observado"]]
    return [
        Ruta(
            id_ruta=int(id_ruta),
            nombre=estado["nombres"].get(int(id_ruta)),
            primera_franja=g["ts"].min(),
            ultima_franja=g["ts"].max(),
            franjas_con_servicio=len(g),
        )
        for id_ruta, g in obs.groupby("id_ruta")
    ]


@app.get("/prediccion/{id_ruta}", response_model=RespuestaPrediccion)
def prediccion(
    id_ruta: int,
    referencia: datetime | None = Query(
        None,
        description="Ultima franja con datos conocidos, en hora local. Por defecto, la ultima "
        "franja con datos de la ruta. Se redondea hacia abajo a 15 min.",
        examples=["2023-02-20T17:00:00"],
    ),
):
    """Subidas previstas en la ruta a 30, 60 y 120 minutos de la referencia."""
    serie = estado["serie"]
    ruta = serie[serie["id_ruta"] == id_ruta]
    if ruta.empty:
        raise HTTPException(404, f"La ruta {id_ruta} no tiene datos. Consulte GET /rutas.")
    ref = _normalizar_referencia(referencia, ruta[ruta["observado"]])

    recientes = ruta[
        ruta["observado"]
        & (ruta["ts"] <= ref)
        & (ruta["ts"] > ref - pd.Timedelta(hours=HORAS_MINIMAS_HISTORIA))
    ]
    if recientes.empty:
        raise HTTPException(
            422,
            f"No hay datos de la ruta {id_ruta} en las {HORAS_MINIMAS_HISTORIA} h previas a "
            f"{ref}; la prediccion no seria confiable. Consulte GET /rutas.",
        )

    hist = _serie_conocida(ruta, ref, ref + pd.Timedelta(minutes=max(estado["modelos"])))
    reales = ruta.set_index("ts")["subidas"]

    predicciones = []
    for h in sorted(estado["modelos"]):
        modelo, meta = estado["modelos"][h]
        objetivo = ref + pd.Timedelta(minutes=h)
        feats = md.calcular_features(hist, h, estado["calendario"])
        fila = feats[feats["ts"] == objetivo]
        predicciones.append(
            Prediccion(
                horizonte_min=h,
                franja_objetivo=objetivo,
                subidas_predichas=round(float(modelo.predict(fila[meta["features"]])[0]), 2),
                linea_base_prom_4_semanas=_o_nada(fila["prom_4_semanas"].iloc[0]),
                subidas_reales=_o_nada(reales.get(objetivo)),
                servicio_habitual=bool(fila["prom_4_semanas"].notna().iloc[0]),
                model_id=meta["model_id"],
            )
        )

    return RespuestaPrediccion(
        id_ruta=id_ruta,
        nombre_ruta=estado["nombres"].get(id_ruta),
        referencia=ref,
        feature_version=md.FEATURE_VERSION,
        predicciones=predicciones,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
