# =================================================================================
# Se Descarga ETL desde el servicio de AWS para versionarlo y subirlo a github 
# =================================================================================
import pandas as pd
import boto3

# =====================================================================
# PASO 1: EXTRACCIÓN (E)
# =====================================================================
path = "s3://proyectointegradorr/entrada_produccion/production_petroleo_2021_2025.csv"
df = pd.read_csv(path)

# Limpieza de nombres de columnas (eliminar espacios en blanco inducidos)
df.columns = df.columns.str.strip()

print("Información general del dataset original:")
df.info()

print("\nTipo de datos sin transformación:")
print(df.dtypes)


# =====================================================================
# PASO 2: SELECCIÓN DE COLUMNAS
# =====================================================================
columnas = [
    "Downtime_flag",
    "Lift_change_flag",
    "Paraffin_inhibitor_ppm",
    "Corrosion_inhibitor_ppm",
    "Water_salinity_ppm",
    "Field_section",
    "Reservoir_pressure_psia",
    "Injection_type",
    "Injection_rate_bwpd",
    "Uptime_pct",
    "Water_rate_bbl_d",
    "Gas_rate_mscf_d",
    "Oil_rate_bbl_d",
    "WaterCut_pct",
    "CumOil_bbl",
    "CumGas_mscf",
    "CumWater_bbl",
    "Latitude",
    "Longitude",
    "Tubing_temp_F",
    "Scale_index",
    "Sand_prod_lbs_d",
    "Lift_system",
    "Well_Type",
    "Date",
    "Well_ID"
]

df_filtrado = df[columnas].copy()

print("\nColumnas seleccionadas:")
print(df_filtrado.dtypes)


# =====================================================================
# PASO 2.1: DATA QUALITY - LIMPIEZA DE REGISTROS EN WELL_TYPE
# =====================================================================
valores_invalidos = ["|-01", "|-02", "|-03", "|-04", "|-05", "|-06"]

# Filtrar y eliminar filas con códigos residuales/errores
df_filtrado = df_filtrado[~df_filtrado["Well_Type"].isin(valores_invalidos)]

print("\nValores únicos de Well_Type después de limpieza:")
print(df_filtrado["Well_Type"].unique())

# =====================================================================
# PASO 2.2: DATA QUALITY - DETECCIÓN Y ELIMINACIÓN DE DUPLICADOS
# =====================================================================
# Definir la clave primaria lógica: un pozo no puede tener dos registros en la misma fecha
print("\nVerificando registros duplicados...")
duplicados_conteo = df_filtrado.duplicated(subset=["Date", "Well_ID"]).sum()
print(f"Cantidad de registros duplicados detectados: {duplicados_conteo}")

if duplicados_conteo > 0:
    # Eliminar duplicados manteniendo solo el primer registro encontrado
    df_filtrado = df_filtrado.drop_duplicates(subset=["Date", "Well_ID"], keep="first")
    print("Duplicados eliminados exitosamente.")


# =====================================================================
# PASO 3: TRANSFORMACIÓN (T) - TIPOS DE DATOS Y FECHAS
# =====================================================================
# Convertir variables categóricas a tipo String explícito
df_filtrado["Field_section"] = df_filtrado["Field_section"].astype("string")
df_filtrado["Injection_type"] = df_filtrado["Injection_type"].astype("string")
df_filtrado["Lift_system"] = df_filtrado["Lift_system"].astype("string")
df_filtrado["Well_Type"] = df_filtrado["Well_Type"].astype("string")

# Convertir columna temporal y remover registros rotos (NaT)
df_filtrado["Date"] = pd.to_datetime(df_filtrado["Date"], errors="coerce")
df_filtrado = df_filtrado.dropna(subset=["Date"])

print("\nTipos de datos después de transformación:")
print(df_filtrado.dtypes)


# =====================================================================
# PASO 3.1: TRANSFORMACIÓN (T) - CODIFICACIÓN CATEGÓRICA (LABEL ENCODING)
# =====================================================================
# Mapeo numérico para Sistemas de Levantamiento
df_filtrado["Lift_system_num"] = df_filtrado["Lift_system"].map({
    "NaturalFlow": 1,
    "GasLift": 2,
    "ESP": 3,
    "RodPump": 4,
    "Injection": 5
})

# Mapeo numérico para Secciones del Campo
df_filtrado["Field_section_num"] = df_filtrado["Field_section"].map({
    "Central": 6,
    "East": 7,
    "North": 8,
    "South": 9,
    "West": 10
})

print("\nTipos de datos después de conversión categórica:")
print(df_filtrado.dtypes)

# Validar posibles nulos generados por el encoding
print("\nNulos después del encoding categórico:")
print(df_filtrado[["Lift_system_num", "Field_section_num"]].isnull().sum())


# =====================================================================
# PASO 4: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =====================================================================
# --- Variables numéricas ---
print("\nResumen estadístico (numéricas):")
numericas = df_filtrado.select_dtypes(include=["number"])
eda = numericas.describe().T
eda["median"] = numericas.median()
print(eda[["count", "mean", "median", "min", "max", "std"]])

# --- Variables categóricas ---
print("\nResumen estadístico (categóricas):")
print(df_filtrado.select_dtypes(include=["string"]).describe())

# --- Auditoría de Valores Nulos ---
print("\nValores nulos por columna:")
print(df_filtrado.isnull().sum())


# =====================================================================
# PASO 5: PROCESAMIENTO ANALÍTICO (TENDENCIAS Y OUTLIERS)
# =====================================================================

# --- 5.1: Análisis de Tendencia Temporal ---
print("\nTendencia de producción de petróleo:")
df_filtrado = df_filtrado.sort_values("Date")

# Tendencia promedio diaria
tendencia = df_filtrado.groupby("Date")["Oil_rate_bbl_d"].mean()
print(tendencia.head())

# Tendencia móvil (Suavizado a 7 días)
print("\nTendencia móvil (7 días):")
tendencia_suave = tendencia.rolling(window=7).mean()
print(tendencia_suave.head())

# --- 5.2: Detección de Outliers - Método IQR ---
print("\nDetección de valores atípicos (IQR):")
Q1 = df_filtrado["Oil_rate_bbl_d"].quantile(0.25)
Q3 = df_filtrado["Oil_rate_bbl_d"].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

outliers_iqr = df_filtrado[
    (df_filtrado["Oil_rate_bbl_d"] < limite_inferior) |
    (df_filtrado["Oil_rate_bbl_d"] > limite_superior)
]
print(outliers_iqr[["Date", "Oil_rate_bbl_d"]].head())

# --- 5.3: Detección de Outliers - Método Z-Score ---
print("\nMétodo Z-score:")
z_score = (df_filtrado["Oil_rate_bbl_d"] - df_filtrado["Oil_rate_bbl_d"].mean()) / df_filtrado["Oil_rate_bbl_d"].std()

outliers_z = df_filtrado[z_score.abs() > 3]
print(outliers_z[["Date", "Oil_rate_bbl_d"]].head())


# =====================================================================
# PASO 6: CARGA (L) - EXPORTACIÓN A STORAGE S3
# =====================================================================
print("\nExportando a S3...")

# Ruta temporal local para entornos serverless (AWS Glue / Lambda)
ruta_local = "/tmp/produccion_limpia.csv"

# Generar archivo físico CSV local
df_filtrado.to_csv(ruta_local, index=False)

# Inicializar cliente SDK boto3 y cargar a bucket destino
s3 = boto3.client("s3")
s3.upload_file(
    ruta_local,
    "proyectointegradorr",
    "Salida_produccion/produccion_limpia.csv"
)

print("Archivo subido correctamente a S3")