"""


Fuente: las lecturas de los sensores de pasajeros (conteo_pasajeros_destino)
de 2023 a 2025, agregadas por ruta y franja de 15 min segun la hora real de
cada subida. Las descarga extraer_datos.py a datos/demanda_15min.parquet.
(Antes se usaba timbradas_segmentos: solo cubre 8 semanas de 2023 y sus horas
estan corridas 2-3 h respecto de los sensores y los despachos; ver eda.ipynb.)

Modos de ejecucion:
  - Con datos/demanda_15min.parquet: datos reales (corra antes extraer_datos.py).
  - Sin el: datos sinteticos con la misma forma, para probar el pipeline.

Que hace:
  1. Carga subidas por ruta y franja de 15 min y aplica las reglas de calidad
     del EDA: dias sin lecturas del sensor (el ultimo de cada mes) y franjas
     con lecturas imposibles quedan como faltantes, no como cero.
  2. Arma una grilla regular de 15 min por ruta.
  3. Para cada horizonte (30, 60, 120 min, RF-005) construye variables que
     solo usan informacion disponible al momento de predecir (seccion 9.3):
     calendario, demanda reciente, estacionalidad semanal y anual, nivel de
     la ruta y oferta de viajes. El detalle esta en feature_engineering.ipynb.
  4. Compara XGBoost contra dos lineas base (seccion 9.1).
  5. Reporta MAE, RMSE, WAPE y sMAPE, global, por ruta y por tipo de dia.
  6. Calcula importancia de variables con SHAP (seccion 9.4).
  7. Guarda cada modelo con sus metadatos en la carpeta modelos/ (seccion 9.2).

Los modelos guardados se sirven con api_demanda.py (FastAPI).

Pendiente (siguientes escalones): nivel segmento y sentido, intervalos de
confianza.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

RANDOM_SEED = 42
PASO_MIN = 15                        # franjas de 15 minutos
PASOS_DIA = 24 * 60 // PASO_MIN      # 96 franjas por dia
PASOS_SEMANA = 7 * PASOS_DIA         # 672 franjas por semana
PASOS_ANIO = 52 * PASOS_SEMANA       # 364 dias: mismo dia de la semana un anio antes
HORIZONTES_MIN = [30, 60, 120]       # RF-005
FEATURE_VERSION = "v3-sensores-3anios"

CARPETA_DATOS = Path(__file__).resolve().parent / "datos"
ARCHIVO_DEMANDA = CARPETA_DATOS / "demanda_15min.parquet"

# Reglas de calidad (ver eda.ipynb, seccion 2)
MIN_FRACCION_LECTURAS_DIA = 0.05     # dia con < 5% de las lecturas habituales -> sin datos
MAX_SUBIDAS_LECTURA = 90             # una lectura (~1 min) no puede superar la capacidad del bus

FEATURES = [
    # calendario (se conoce de antemano)
    "id_ruta", "minuto_dia", "dia_semana", "mes", "dia_del_anio",
    "tipo_dia", "es_festivo", "precarnaval", "carnaval_central", "semana_santa", "impacto_evento",
    # demanda reciente (hasta la franja de referencia t - h)
    "lag_h", "lag_h_mas_1", "media_1h_previa", "ratio_hoy", "nivel_28d",
    # estacionalidad semanal y anual
    "lag_1d", "lag_7d", "prom_4_semanas", "lag_364d",
    # oferta de viajes: la reciente y la habitual (la futura no se conoce, ver EDA)
    "viajes_1h_previa", "viajes_habituales",
]


# ---------------------------------------------------------------------------
# 1. CARGA Y LIMPIEZA DE DATOS
# ---------------------------------------------------------------------------
def cargar_datos() -> pd.DataFrame:
    if ARCHIVO_DEMANDA.exists():
        return limpiar_demanda(pd.read_parquet(ARCHIVO_DEMANDA))
    print(f"(no existe {ARCHIVO_DEMANDA.name} -> datos sinteticos; para los reales corra extraer_datos.py)")
    return _generar_datos_sinteticos()


def limpiar_demanda(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Reglas de validez (seccion 6.3) encontradas en el EDA."""
    df = df.copy()
    df["fecha"] = df["ts"].dt.normalize()

    # 1. Dias sin lecturas del sensor: el ultimo dia de cada mes la tabla de
    #    sensores esta casi vacia aunque los viajes si se hicieron.
    lect_dia = df.groupby(["id_ruta", "fecha"])["lecturas"].transform("sum")
    mediana = df.groupby(["id_ruta", "fecha"])["lecturas"].sum().groupby("id_ruta").median()
    sin_datos = lect_dia < MIN_FRACCION_LECTURAS_DIA * df["id_ruta"].map(mediana)

    # 2. Lecturas imposibles (saltos del contador): se descarta la franja entera
    imposibles = df["max_subida_lectura"] > MAX_SUBIDAS_LECTURA

    if verbose:
        dias = df.loc[sin_datos, ["id_ruta", "fecha"]].drop_duplicates()
        print(f"Limpieza: {len(dias)} dias-ruta sin lecturas del sensor y "
              f"{int((imposibles & ~sin_datos).sum())} franjas con lecturas imposibles -> faltantes")
    return df[~sin_datos & ~imposibles].drop(columns="fecha").reset_index(drop=True)


def _crear_engine():
    from sqlalchemy import create_engine
    from sqlalchemy.engine import URL

    password = os.environ.get("SOBUSA_DB_PASSWORD") or os.environ.get("PGPASSWORD")
    if not password:
        raise RuntimeError(
            "Falta la contrasena: defina SOBUSA_DB_PASSWORD o PGPASSWORD "
            "(si la guardo con setx, reinicie la terminal o VS Code)."
        )
    url = URL.create(
        "postgresql+psycopg2",
        username=os.environ["SOBUSA_DB_USER"],
        password=password,
        host=os.environ["SOBUSA_DB_HOST"],
        port=int(os.environ.get("SOBUSA_DB_PORT", "5432")),
        database=os.environ.get("SOBUSA_DB_NAME", "dataset"),
    )
    # Solo lectura + timeout de 5 minutos por consulta (seccion 11.3)
    return create_engine(
        url,
        connect_args={
            "options": "-c default_transaction_read_only=on -c statement_timeout=300000",
            "sslmode": os.environ.get("SOBUSA_DB_SSLMODE", "prefer"),
        },
    )


QUERY_ESTACIONALIDADES = """
    SELECT e.estacionalidad, ea.id_estacionalidad, ea.fecha_inicio, ea.fecha_fin
    FROM public.estacionalidades_anio ea
    JOIN public.estacionalidades e USING (id_estacionalidad)
    WHERE e.estacionalidad <> 'DIAS_CORRIENTES'
"""

QUERY_NOVEDADES = """
    SELECT nf.fecha, nf.hora_inicio, nf.hora_fin, nf.porcentaje_impacto, n.nombre AS evento
    FROM public.novedades_viales_fecha nf
    JOIN public.novedades_viales n USING (id_novedad_vial)
"""

ID_PRECARNAVAL = 7         # estacionalidades: Precarnaval y Carnaval de Barranquilla
ID_CARNAVAL_CENTRAL = 8    # estacionalidades: Carnaval - dias centrales
ID_SEMANA_SANTA = 10       # estacionalidades: Semana Santa


def cargar_calendario():
    """Festivos, temporadas y eventos: se conocen de antemano, asi que usarlos
    como variables no es fuga temporal. Se leen de datos/ (extraer_datos.py);
    si no estan, de la base; sin ninguna de las dos, calendario vacio."""
    if (CARPETA_DATOS / "estacionalidades.parquet").exists():
        estac = pd.read_parquet(CARPETA_DATOS / "estacionalidades.parquet")
        novedades = pd.read_parquet(CARPETA_DATOS / "novedades.parquet")
        for col in ["hora_inicio", "hora_fin"]:
            novedades[col] = pd.to_datetime(novedades[col], format="%H:%M:%S").dt.time
        return estac, novedades
    if os.environ.get("SOBUSA_DB_HOST"):
        engine = _crear_engine()
        with engine.connect() as conn:
            return pd.read_sql(QUERY_ESTACIONALIDADES, conn), pd.read_sql(QUERY_NOVEDADES, conn)
    return (
        pd.DataFrame(columns=["estacionalidad", "id_estacionalidad", "fecha_inicio", "fecha_fin"]),
        pd.DataFrame(columns=["fecha", "hora_inicio", "hora_fin", "porcentaje_impacto"]),
    )


def _generar_datos_sinteticos() -> pd.DataFrame:
    """Misma forma que los datos reales: id_ruta, ts, subidas, bajadas,
    nro_viajes. Servicio de 04:30 a 22:00."""
    rng = np.random.default_rng(RANDOM_SEED)
    dias = pd.date_range("2023-01-01", "2025-06-30", freq="D")
    franjas = pd.timedelta_range("04:30:00", "21:45:00", freq=f"{PASO_MIN}min")
    minutos = franjas.total_seconds().values / 60

    # Perfil diario con pico de manana (7:00) y de tarde (17:30)
    perfil = (
        4
        + 30 * np.exp(-((minutos - 420) ** 2) / (2 * 60 ** 2))
        + 24 * np.exp(-((minutos - 1050) ** 2) / (2 * 75 ** 2))
    )

    partes = []
    for id_ruta, escala in [(2, 1.0), (10, 1.4), (11, 0.8)]:
        # Nivel diario que deriva lentamente: hace utiles los rezagos
        nivel = np.exp(np.cumsum(rng.normal(0, 0.03, len(dias))))
        nivel = nivel / nivel.mean()
        for i, dia in enumerate(dias):
            factor = escala * nivel[i] * (0.6 if dia.dayofweek >= 5 else 1.0)
            partes.append(
                pd.DataFrame(
                    {
                        "id_ruta": id_ruta,
                        "ts": dia + franjas,
                        "subidas": rng.poisson(perfil * factor),
                        "bajadas": rng.poisson(perfil * factor * 0.95),
                        "nro_viajes": 3,
                    }
                )
            )
    return pd.concat(partes, ignore_index=True)


# ---------------------------------------------------------------------------
# 2. SERIE REGULAR DE 15 MINUTOS POR RUTA (+ validacion basica, seccion 6.3)
# ---------------------------------------------------------------------------
def preparar_serie(crudo: pd.DataFrame) -> pd.DataFrame:
    df = crudo.copy()
    df["ts"] = pd.to_datetime(df["ts"]).dt.floor(f"{PASO_MIN}min")

    negativos = int((df["subidas"] < 0).sum())
    if negativos:
        print(f"AVISO: {negativos} filas con subidas negativas, se excluyen (regla de validez 6.3).")
        df = df[df["subidas"] >= 0]

    df = df.groupby(["id_ruta", "ts"], as_index=False)[
        ["subidas", "bajadas", "nro_viajes"]
    ].sum()

    partes = []
    for id_ruta, g in df.groupby("id_ruta"):
        inicio = g["ts"].min().normalize()
        fin = g["ts"].max().normalize() + pd.Timedelta(days=1) - pd.Timedelta(minutes=PASO_MIN)
        grilla = pd.date_range(inicio, fin, freq=f"{PASO_MIN}min")
        g = g.set_index("ts").reindex(grilla).rename_axis("ts").reset_index()
        g["id_ruta"] = id_ruta
        g["observado"] = g["subidas"].notna()  # sin servicio o sin dato -> faltante, no cero
        partes.append(g)
    return pd.concat(partes, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. VARIABLES POR HORIZONTE (sin fuga temporal: todo usa datos de t - h o antes)
# ---------------------------------------------------------------------------
def _dias_en_rangos(rangos: pd.DataFrame) -> set:
    dias = set()
    for ini, fin in zip(rangos["fecha_inicio"], rangos["fecha_fin"]):
        dias.update(pd.date_range(ini, fin, freq="D"))
    return dias


def agregar_calendario(df: pd.DataFrame, calendario) -> pd.DataFrame:
    estac, novedades = calendario
    dia = df["ts"].dt.normalize()
    # Solo lo que cae dentro de la serie: mismo resultado, mucho mas rapido en el servicio
    d0, d1 = dia.min(), dia.max()
    estac = estac[(pd.to_datetime(estac["fecha_fin"]) >= d0) & (pd.to_datetime(estac["fecha_inicio"]) <= d1)]
    novedades = novedades[pd.to_datetime(novedades["fecha"]).between(d0, d1)]

    festivos = _dias_en_rangos(estac[estac["estacionalidad"] == "FESTIVO_NACIONAL"])
    precarnaval = _dias_en_rangos(estac[estac["id_estacionalidad"] == ID_PRECARNAVAL])
    carnaval = _dias_en_rangos(estac[estac["id_estacionalidad"] == ID_CARNAVAL_CENTRAL])
    semana_santa = _dias_en_rangos(estac[estac["id_estacionalidad"] == ID_SEMANA_SANTA])
    df["es_festivo"] = dia.isin(festivos).astype(int)
    df["precarnaval"] = dia.isin(precarnaval).astype(int)
    df["carnaval_central"] = dia.isin(carnaval).astype(int)
    df["semana_santa"] = dia.isin(semana_santa).astype(int)

    # 0 = habil, 1 = sabado, 2 = domingo o festivo. El lunes y martes de
    # Carnaval se comportan como festivo en Barranquilla, asi que tambien
    # cuentan como tipo 2 (ver eda.ipynb, seccion 6).
    tipo = np.select([df["dia_semana"] == 5, df["dia_semana"] == 6], [1, 2], 0)
    tipo = np.where(df["es_festivo"].eq(1), 2, tipo)
    tipo = np.where(df["carnaval_central"].eq(1) & (df["dia_semana"] < 5), 2, tipo)
    df["tipo_dia"] = tipo

    # Eventos (Guacherna, Batalla de Flores, partidos...): % de impacto en su ventana horaria
    df["impacto_evento"] = 0.0
    hora = df["ts"].dt.time
    for _, n in novedades.iterrows():
        activo = (dia == pd.Timestamp(n["fecha"])) & (hora >= n["hora_inicio"]) & (hora <= n["hora_fin"])
        df.loc[activo, "impacto_evento"] = np.maximum(
            df.loc[activo, "impacto_evento"], float(n["porcentaje_impacto"] or 0)
        )
    return df


def calcular_features(serie: pd.DataFrame, horizonte_min: int, calendario) -> pd.DataFrame:
    """Variables de todas las franjas de la grilla, con o sin servicio. La usan
    el entrenamiento y el servicio FastAPI (api_demanda.py), asi ambos calculan
    exactamente igual. Para la franja t con horizonte h, lo observado solo se
    toma hasta t - h (la referencia): shift(h) o mas."""
    h = horizonte_min // PASO_MIN
    df = serie.sort_values(["id_ruta", "ts"]).copy()

    df["minuto_dia"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df["dia_semana"] = df["ts"].dt.dayofweek
    df["mes"] = df["ts"].dt.month
    df["dia_del_anio"] = df["ts"].dt.dayofyear
    df = agregar_calendario(df, calendario)

    g = df.groupby("id_ruta")["subidas"]
    df["lag_h"] = g.shift(h)                  # ultimo valor conocido al predecir
    df["lag_h_mas_1"] = g.shift(h + 1)
    df["media_1h_previa"] = g.transform(lambda s: s.shift(h).rolling(4, min_periods=1).mean())
    df["lag_1d"] = g.shift(PASOS_DIA)         # 96 >= h para los tres horizontes
    df["lag_7d"] = g.shift(PASOS_SEMANA)
    semanas = [g.shift(PASOS_SEMANA * k) for k in range(1, 5)]
    df["prom_4_semanas"] = pd.concat(semanas, axis=1).mean(axis=1)
    df["lag_364d"] = g.shift(PASOS_ANIO)      # estacionalidad anual, mismo dia de la semana

    # Nivel de la ruta: subidas medias por franja con servicio en los 28 dias
    # previos a la referencia (sigue la caida de ~7% anual que muestra el EDA)
    df["nivel_28d"] = g.transform(
        lambda s: s.shift(h).rolling(28 * PASOS_DIA, min_periods=PASOS_DIA).mean()
    )

    # Como va el dia: subidas acumuladas del dia hasta la referencia frente a
    # lo habitual a esa misma hora (promedio de las 4 semanas previas). Capta
    # dias atipicos que el calendario no anuncia (lluvia, paros, eventos).
    acum = df.assign(_s=df["subidas"].fillna(0)).groupby(["id_ruta", df["ts"].dt.normalize()])["_s"].cumsum()
    ga = acum.groupby(df["id_ruta"])
    acum_ref = ga.shift(h)
    acum_habitual = pd.concat([ga.shift(h + PASOS_SEMANA * k) for k in range(1, 5)], axis=1).mean(axis=1)
    df["ratio_hoy"] = (acum_ref / acum_habitual).where(acum_habitual >= 50)

    # Oferta: viajes con lecturas en la ultima hora conocida y los habituales en
    # la franja objetivo. Los viajes programados de la franja objetivo NO se
    # usan: se registran ~12 min antes de salir, no se conocen a 30-120 min.
    gv = df.groupby("id_ruta")["nro_viajes"]
    df["viajes_1h_previa"] = gv.transform(lambda s: s.shift(h).rolling(4, min_periods=1).mean())
    df["viajes_habituales"] = pd.concat([gv.shift(PASOS_SEMANA * k) for k in range(1, 5)], axis=1).mean(axis=1)

    df["tiene_base"] = df["lag_7d"].notna() & df["prom_4_semanas"].notna()
    return df


def construir_features(serie: pd.DataFrame, horizonte_min: int, calendario) -> pd.DataFrame:
    # Se entrena en toda franja con servicio (XGBoost tolera rezagos faltantes).
    # Se evalua solo donde existen ambas lineas base.
    df = calcular_features(serie, horizonte_min, calendario)
    return df[df["observado"]].reset_index(drop=True)


def particion_temporal(df: pd.DataFrame, frac_val=0.1, frac_test=0.2):
    """Cortes cronologicos calculados sobre el rango con lineas base."""
    tiempos = np.sort(df.loc[df["tiene_base"], "ts"].unique())
    corte_val = tiempos[int(len(tiempos) * (1 - frac_val - frac_test))]
    corte_test = tiempos[int(len(tiempos) * (1 - frac_test))]
    train = df[df["ts"] < corte_val]
    val = df[(df["ts"] >= corte_val) & (df["ts"] < corte_test) & df["tiene_base"]]
    test = df[(df["ts"] >= corte_test) & df["tiene_base"]]
    return train, val, test, pd.Timestamp(corte_val), pd.Timestamp(corte_test)


def tipo_periodo(df: pd.DataFrame) -> np.ndarray:
    """Dias especiales (festivo, Carnaval, Semana Santa) frente a dias normales."""
    especial = df["es_festivo"].eq(1) | df["carnaval_central"].eq(1) | df["semana_santa"].eq(1)
    return np.where(especial, "especial", "normal")


# ---------------------------------------------------------------------------
# 4. METRICAS (seccion 9.3)
# ---------------------------------------------------------------------------
def metricas(y, yhat) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return {
        "MAE": float(np.mean(np.abs(y - yhat))),
        "RMSE": float(np.sqrt(np.mean((y - yhat) ** 2))),
        "WAPE": float(np.sum(np.abs(y - yhat)) / max(np.sum(np.abs(y)), 1e-9)),
        "sMAPE": float(np.mean(2 * np.abs(yhat - y) / (np.abs(y) + np.abs(yhat) + 1e-9))),
    }


def fila(nombre, m):
    return (
        f"  {nombre:16s} | MAE {m['MAE']:6.2f} | RMSE {m['RMSE']:6.2f} "
        f"| WAPE {m['WAPE']:6.1%} | sMAPE {m['sMAPE']:6.1%}"
    )


def nuevo_modelo():
    import xgboost as xgb

    return xgb.XGBRegressor(
        objective="count:poisson",   # la demanda es un conteo >= 0
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,    # usa val; test queda intacto
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import shap

    crudo = cargar_datos()
    calendario = cargar_calendario()
    serie = preparar_serie(crudo)
    obs = serie[serie["observado"]]
    print(
        f"Franjas con servicio: {len(obs):,} | Rutas: {sorted(obs['id_ruta'].unique().tolist())} "
        f"| Desde {obs['ts'].min():%Y-%m-%d} hasta {obs['ts'].max():%Y-%m-%d}"
    )

    os.makedirs("modelos", exist_ok=True)

    for horizonte in HORIZONTES_MIN:
        df = construir_features(serie, horizonte, calendario)
        train, val, test, c_val, c_test = particion_temporal(df)
        print(f"\n=== Horizonte {horizonte} min ===")
        print(
            f"  Train {len(train):,} | Val {len(val):,} (desde {c_val:%Y-%m-%d}) "
            f"| Test {len(test):,} (desde {c_test:%Y-%m-%d})"
        )

        modelo = nuevo_modelo()
        modelo.fit(
            train[FEATURES], train["subidas"],
            eval_set=[(val[FEATURES], val["subidas"])], verbose=False,
        )

        y = test["subidas"]
        pred = modelo.predict(test[FEATURES])
        res = {
            "Base: sem. pasada": metricas(y, test["lag_7d"]),
            "Base: prom 4 sem": metricas(y, test["prom_4_semanas"]),
            "XGBoost": metricas(y, pred),
        }
        print("  Prueba (datos que el modelo nunca vio):")
        for nombre, m in res.items():
            print(fila(nombre, m))

        print("  WAPE por ruta (base prom 4 sem -> XGBoost):")
        for id_ruta, g in test.assign(pred=pred).groupby("id_ruta"):
            b = metricas(g["subidas"], g["prom_4_semanas"])["WAPE"]
            x = metricas(g["subidas"], g["pred"])["WAPE"]
            print(f"    ruta {id_ruta}: {b:6.1%} -> {x:6.1%}")

        print("  WAPE por tipo de dia (base prom 4 sem -> XGBoost):")
        for nombre, g in test.assign(pred=pred, periodo=tipo_periodo(test)).groupby("periodo"):
            b = metricas(g["subidas"], g["prom_4_semanas"])["WAPE"]
            x = metricas(g["subidas"], g["pred"])["WAPE"]
            print(f"    {nombre:8s}: {b:6.1%} -> {x:6.1%}  ({len(g):,} franjas)")

        muestra = test[FEATURES].sample(min(2000, len(test)), random_state=RANDOM_SEED)
        shap_vals = shap.TreeExplainer(modelo).shap_values(muestra)
        importancia = pd.Series(np.abs(shap_vals).mean(axis=0), index=FEATURES)
        top = importancia.sort_values(ascending=False).head(5)
        print("  Top 5 variables (SHAP): " + ", ".join(top.index))

        # Artefacto + metadatos (seccion 9.2)
        base = f"modelos/demanda_h{horizonte}"
        modelo.save_model(f"{base}.json")
        with open(f"{base}_meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_id": f"demanda-xgb-h{horizonte}",
                    "feature_version": FEATURE_VERSION,
                    "features": FEATURES,
                    "horizonte_min": horizonte,
                    "data_cutoff": str(df["ts"].max()),
                    "corte_validacion": str(c_val),
                    "corte_prueba": str(c_test),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "semilla": RANDOM_SEED,
                    "fuente": "sensores" if ARCHIVO_DEMANDA.exists() else "sintetica",
                    "metricas_prueba": res,
                },
                f, ensure_ascii=False, indent=2,
            )

    print("\nModelos y metadatos guardados en la carpeta modelos/")
