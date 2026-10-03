"""
Extraccion de TODO el historico disponible - SOBUSA Data Challenge 2026.

timbradas_segmentos solo cubre 2023-01-01 a 2023-02-27. La base guarda mucho
mas: los sensores de pasajeros (conteo_pasajeros_destino, ~176 millones de
lecturas) y la programacion de viajes (progvehiculos) de 2023 a 2025.

Este script arma una copia local en datos/ para que el EDA y el entrenamiento
no dependan de consultas pesadas:
  - viajes.parquet          un registro por viaje programado (progvehiculos)
  - demanda_15min.parquet   subidas y bajadas por ruta y franja de 15 min,
                            desde las lecturas de los sensores (hora real de
                            cada subida, no la hora de salida del bus)
  - estacionalidades.parquet, novedades.parquet   calendario
  - rutas.parquet           nombre de cada ruta

conteo_pasajeros_destino no tiene indice por fecha, solo por idprogramacion:
por eso se consulta mes a mes con la lista de viajes del mes, agregando en el
servidor. Cada mes se guarda aparte (datos/conteo_mensual/), asi que si se
corta se puede volver a correr y retoma donde iba.

Uso:  python extraer_datos.py        (tarda del orden de 15-30 minuto)
"""

import time
from pathlib import Path

import pandas as pd
from sqlalchemy import text

import modelo_demanda as md

CARPETA = Path(__file__).resolve().parent / "datos"
CARPETA_MES = CARPETA / "conteo_mensual"

QUERY_VIAJES = """
    SELECT idprogramacion, idruta AS id_ruta, idvehiculo, fechasalida, fecharegistro,
           fechainicio, fechafin, cancelada, finalizado, numeropasajeros, numerobajadas,
           distancia_recorrida, excedeplazosalida
    FROM public.progvehiculos
"""

# Subidas por la hora real de cada lectura del sensor. Las lecturas con
# cantidad negativa (reinicios del contador) no suman: se cuentan aparte.
QUERY_CONTEO = """
    SELECT p.idruta                                        AS id_ruta,
           date_bin('15 minutes', c.fecha, TIMESTAMP '2000-01-01') AS ts,
           SUM(GREATEST(c.cantidad_subida, 0))             AS subidas,
           SUM(GREATEST(c.cantidad_bajada, 0))             AS bajadas,
           COUNT(DISTINCT c.idprogramacion)                AS nro_viajes,
           COUNT(DISTINCT c.idvehiculo)                    AS nro_vehiculos,
           COUNT(*)                                        AS lecturas,
           SUM(CASE WHEN c.cantidad_subida < 0 THEN 1 ELSE 0 END) AS lecturas_negativas,
           MAX(c.cantidad_subida)                          AS max_subida_lectura
    FROM public.conteo_pasajeros_destino c
    JOIN public.progvehiculos p ON p.idprogramacion = c.idprogramacion
    WHERE c.idprogramacion = ANY(:ids)
    GROUP BY 1, 2
"""


def _consultar_conteo(conn, ids: list) -> pd.DataFrame:
    """Si una consulta excede el timeout, parte la lista en dos y reintenta."""
    try:
        return pd.read_sql(text(QUERY_CONTEO), conn, params={"ids": ids})
    except Exception as e:
        if "statement timeout" not in str(e) or len(ids) < 200:
            raise
        conn.rollback()
        mitad = len(ids) // 2
        print(f"    timeout con {len(ids)} viajes, se parte en dos", flush=True)
        return pd.concat([_consultar_conteo(conn, ids[:mitad]), _consultar_conteo(conn, ids[mitad:])])


def main():
    CARPETA_MES.mkdir(parents=True, exist_ok=True)
    engine = md._crear_engine()

    print("Viajes (progvehiculos)...", flush=True)
    with engine.connect() as conn:
        viajes = pd.read_sql(text(QUERY_VIAJES), conn)
    viajes.to_parquet(CARPETA / "viajes.parquet", index=False)
    print(f"  {len(viajes):,} viajes, {viajes['fechasalida'].min()} a {viajes['fechasalida'].max()}", flush=True)

    with engine.connect() as conn:
        pd.read_sql(text(md.QUERY_ESTACIONALIDADES), conn).to_parquet(CARPETA / "estacionalidades.parquet", index=False)
        pd.read_sql(text(md.QUERY_NOVEDADES), conn).astype({"hora_inicio": str, "hora_fin": str}).to_parquet(
            CARPETA / "novedades.parquet", index=False)
        pd.read_sql(text("SELECT idruta AS id_ruta, nombre FROM public.rutas"), conn).to_parquet(
            CARPETA / "rutas.parquet", index=False)

    meses = viajes["fechasalida"].dt.to_period("M")
    for mes in sorted(meses.unique()):
        archivo = CARPETA_MES / f"{mes}.parquet"
        if archivo.exists():
            continue
        ids = viajes.loc[meses == mes, "idprogramacion"].astype(int).tolist()
        t0 = time.perf_counter()
        with engine.connect() as conn:
            df = _consultar_conteo(conn, ids)
        df.to_parquet(archivo, index=False)
        print(f"  {mes}: {len(ids):,} viajes -> {len(df):,} franjas, "
              f"{df['subidas'].sum():,.0f} subidas ({time.perf_counter() - t0:.0f}s)", flush=True)

    # Un viaje que cruza la medianoche de fin de mes deja la misma franja en dos
    # archivos: se vuelve a agregar. (nro_viajes/vehiculos pueden sobrecontar 1
    # en esas pocas franjas.)
    partes = pd.concat([pd.read_parquet(a) for a in sorted(CARPETA_MES.glob("*.parquet"))])
    demanda = partes.groupby(["id_ruta", "ts"], as_index=False).agg(
        subidas=("subidas", "sum"), bajadas=("bajadas", "sum"),
        nro_viajes=("nro_viajes", "sum"), nro_vehiculos=("nro_vehiculos", "sum"),
        lecturas=("lecturas", "sum"), lecturas_negativas=("lecturas_negativas", "sum"),
        max_subida_lectura=("max_subida_lectura", "max"),
    )
    demanda.to_parquet(CARPETA / "demanda_15min.parquet", index=False)
    print(f"Listo: {len(demanda):,} franjas, {demanda['ts'].min()} a {demanda['ts'].max()}, "
          f"{demanda['subidas'].sum():,.0f} subidas -> {CARPETA}", flush=True)


if __name__ == "__main__":
    main()
