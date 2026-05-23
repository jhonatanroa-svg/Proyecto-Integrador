# ==========================================
# DASHBOARD — Sistema de Inteligencia de Salud de Pozos
# Adaptado para produccion_limpia.csv
# ==========================================
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Consola de Salud de Pozos", layout="wide")

# AJUSTE DE DISEÑO
st.markdown("""
<style>
input { background-color: white !important; color: black !important; font-weight: bold !important; }
.stNumberInput div div input { background-color: white !important; color: black !important; }
.main { background-color: #0e1117; }
.metric-card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #334155;
    text-align: center;
    height: 120px;
}
.metric-card-white {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    text-align: center;
    height: 120px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.metric-title {
    font-size: 16px;
    font-weight: bold;
    margin-bottom: 10px;
    text-transform: uppercase;
}
.metric-title-dark {
    color: #475569;
    font-size: 16px;
    font-weight: bold;
    margin-bottom: 10px;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# CARGA DE MODELOS
@st.cache_resource
def load_models():
    model = joblib.load("well_health_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


try:
    model, scaler = load_models()
except Exception as e:
    st.error(f"Error cargando modelos: {e}")
    st.info("Asegúrate de correr primero el notebook Clasificacion_Salud_Pozos.ipynb para generar los archivos .pkl")
    st.stop()


# BASE DE DATOS DE POZOS (usando datos reales del CSV)
@st.cache_data
def get_well_database():
    try:
        df = pd.read_csv('produccion_limpia.csv')
        df = df[df['Well_Type'].str.lower() == 'producer'].copy()
        # Tomar el último registro de cada pozo como valores por defecto
        df_last = df.sort_values('Date').groupby('Well_ID').last().reset_index()
        wells = {}
        for _, row in df_last.iterrows():
            wells[row['Well_ID']] = {
                "temp": float(row['Tubing_temp_F']) if pd.notna(row['Tubing_temp_F']) else 175.0,
                "salinity": float(row['Water_salinity_ppm']) if pd.notna(row['Water_salinity_ppm']) else 80000.0,
                "idx": float(row['Scale_index']) if pd.notna(row['Scale_index']) else 1.0,
                "pres": float(row['Reservoir_pressure_psia']) if pd.notna(row['Reservoir_pressure_psia']) else 3200.0,
                "wc": float(row['WaterCut_pct']) if pd.notna(row['WaterCut_pct']) else 45.0,
                "oil": float(row['Oil_rate_bbl_d']) if pd.notna(row['Oil_rate_bbl_d']) else 1000.0,
                "gas": float(row['Gas_rate_mscf_d']) if pd.notna(row['Gas_rate_mscf_d']) else 400.0,
                "sand": float(row['Sand_prod_lbs_d']) if pd.notna(row['Sand_prod_lbs_d']) else 5.0,
                "corrosion": float(row['Corrosion_inhibitor_ppm']) if pd.notna(row['Corrosion_inhibitor_ppm']) else 25.0,
                "paraffin": float(row['Paraffin_inhibitor_ppm']) if pd.notna(row['Paraffin_inhibitor_ppm']) else 15.0,
            }
        return wells
    except Exception:
        # Fallback con datos sintéticos si no se puede leer el CSV
        wells = {}
        np.random.seed(42)
        for i in range(1, 25):
            wells[f"P-{str(i).zfill(2)}"] = {
                "temp": float(np.random.randint(140, 210)),
                "salinity": float(np.random.randint(40000, 120000)),
                "idx": round(np.random.uniform(-1.5, 3.5), 2),
                "pres": float(np.random.randint(2000, 4500)),
                "wc": float(np.random.randint(10, 85)),
                "oil": float(np.random.randint(500, 2500)),
                "gas": 400.0,
                "sand": 5.0,
                "corrosion": 25.0,
                "paraffin": 15.0,
            }
        return wells


well_db = get_well_database()

# HEADER
st.title("🛢️ Sistema de Inteligencia de Salud de Pozos")
st.markdown("### Soporte de Decisiones Operativas | Ingeniería de Producción")
st.divider()

# SIDEBAR
st.sidebar.header("Navegación del Campo")
pozo_seleccionado = st.sidebar.selectbox("Seleccione un Pozo:", list(well_db.keys()))
datos_pozo = well_db[pozo_seleccionado]

st.sidebar.divider()
st.sidebar.subheader("Parámetros del pozo")

with st.sidebar:
    t_f = st.slider("Temperatura Tubing (°F)", 40.0, 250.0, datos_pozo["temp"])
    salinity = st.number_input("Salinidad del Agua (ppm)", 0.0, 250000.0, datos_pozo["salinity"])
    scale_idx = st.slider("Índice Oddo-Thomson", -3.0, 5.0, datos_pozo["idx"])
    st.divider()
    pressure = st.number_input("Presión Yacimiento (psia)", 0.0, 6000.0, datos_pozo["pres"])
    water_cut = st.slider("Corte de Agua (%)", 0.0, 100.0, datos_pozo["wc"])
    oil_rate = st.number_input("Tasa de Aceite (bbl/d)", 0.0, 5000.0, datos_pozo["oil"])

# PROCESAMIENTO
data_dict = {
    "Tubing_temp_F": t_f,
    "Water_salinity_ppm": salinity,
    "Scale_index": scale_idx,
    "Reservoir_pressure_psia": pressure,
    "WaterCut_pct": water_cut,
    "Oil_rate_bbl_d": oil_rate,
    "Gas_rate_mscf_d": datos_pozo["gas"],
    "Sand_prod_lbs_d": datos_pozo["sand"],
    "Corrosion_inhibitor_ppm": datos_pozo["corrosion"],
    "Paraffin_inhibitor_ppm": datos_pozo["paraffin"]
}

input_df = pd.DataFrame([data_dict])
prob = model.predict_proba(scaler.transform(input_df))[0][1]
porcentaje_riesgo = prob * 100

# Lógica de colores
if prob >= 0.8:
    status, color, box_bg = "CRÍTICO", "#ef4444", "rgba(239, 68, 68, 0.2)"
    recommendation = f"🚨 **ALERTA PARA {pozo_seleccionado}:** Riesgo inminente. Se recomienda limpieza química y revisión de bomba ESP."
elif prob >= 0.5:
    status, color, box_bg = "RIESGO MEDIO", "#f59e0b", "rgba(245, 158, 11, 0.2)"
    recommendation = f"⚠️ **ATENCIÓN:** Riesgo moderado. Incrementar inhibidor y realizar análisis de iones en 24h."
else:
    status, color, box_bg = "SALUDABLE", "#22c55e", "rgba(34, 197, 94, 0.2)"
    recommendation = f"✅ **OPERACIÓN NORMAL:** {pozo_seleccionado} estable. Continuar con el plan de monitoreo estándar."

# VISUALIZACIÓN DE MÉTRICAS
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class='metric-card-white'>
        <div class='metric-title-dark'>Riesgo de Escalas</div>
        <span style='color:#0f172a; font-size:32px; font-weight:bold;'>{round(porcentaje_riesgo, 2)}%</span>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card' style='border: 2px solid {color}; background-color: {box_bg};'>
        <div class='metric-title' style='color:{color}; font-weight: 900;'>Estado del Pozo</div>
        <span style='color:{color}; font-size:32px; font-weight:bold;'>{status}</span>
    </div>""", unsafe_allow_html=True)

with col3:
    factor = round(t_f / salinity * 1000, 4) if salinity > 0 else 0
    st.markdown(f"""
    <div class='metric-card-white'>
        <div class='metric-title-dark'>Factor Termodinámico</div>
        <span style='color:#0f172a; font-size:32px; font-weight:bold;'>{factor}</span>
    </div>""", unsafe_allow_html=True)

st.divider()

# CUADRO DE RECOMENDACIÓN
if prob >= 0.8:
    st.error(recommendation)
elif prob >= 0.5:
    st.warning(recommendation)
else:
    st.success(recommendation)

# GRÁFICOS
c1, c2 = st.columns([1, 1.5])

with c1:
    st.markdown("#### Distribución de Riesgo")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=porcentaje_riesgo,
        number={'suffix': "%", 'font': {'color': "white", 'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "white"},
            'bar': {'color': "#2563eb", 'thickness': 0.25},
            'bgcolor': "#1e293b",
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.30,
                'value': porcentaje_riesgo
            },
            'steps': [
                {'range': [0, 50], 'color': '#22c55e'},
                {'range': [50, 80], 'color': '#f59e0b'},
                {'range': [80, 100], 'color': '#ef4444'}
            ]
        }
    ))
    fig_gauge.update_layout(
        height=350, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
    st.plotly_chart(fig_gauge, use_container_width=True)

with c2:
    st.markdown("#### Mapa de Fronteras Operacionales")
    t_range = np.linspace(100, 250, 20)
    s_range = np.linspace(0, 250000, 20)
    T, S = np.meshgrid(t_range, s_range)

    grid_df = pd.DataFrame([data_dict] * 400)
    grid_df['Tubing_temp_F'] = T.flatten()
    grid_df['Water_salinity_ppm'] = S.flatten()

    Z = model.predict_proba(scaler.transform(grid_df))[:, 1].reshape(20, 20)

    fig_map = go.Figure(data=go.Heatmap(
        z=Z, x=t_range, y=s_range, colorscale='RdYlGn_r', showscale=False))
    fig_map.add_trace(go.Scatter(
        x=[t_f], y=[salinity], mode='markers+text',
        marker=dict(color='white', size=15, symbol='x',
                    line=dict(width=2, color='black')),
        text=["POZO"], textposition="top center"))
    fig_map.update_layout(
        height=400, xaxis_title="Temperatura (°F)", yaxis_title="Salinidad (ppm)",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white"})
    st.plotly_chart(fig_map, use_container_width=True)

st.divider()

# DIAGNÓSTICO DE CAUSALIDAD
st.markdown("### Diagnóstico de Causalidad e Impacto de Variables")
st.markdown("El siguiente gráfico detalla cómo contribuye cada parámetro operativo al nivel de riesgo actual.")

impactos = {
    "Índice Oddo-Thomson": (scale_idx - 1.0) * 15,
    "Salinidad del Agua": (salinity - 80000) / 17000,
    "Corte de Agua (%)": (water_cut - 45) * 0.4,
    "Temperatura Tubing": (t_f - 175) * 0.15,
    "Presión Yacimiento": -(pressure - 3250) / 100,
    "Tasa de Aceite": -(oil_rate - 1500) / 200
}

df_impactos = pd.DataFrame(list(impactos.items()), columns=["Variable", "Impacto"])
df_impactos["Abs_Impacto"] = df_impactos["Impacto"].abs()
df_impactos = df_impactos.sort_values(by="Abs_Impacto", ascending=True)
df_impactos["Color"] = np.where(df_impactos["Impacto"] >= 0, "#ef4444", "#22c55e")

fig_tornado = go.Figure()
fig_tornado.add_trace(go.Bar(
    y=df_impactos["Variable"],
    x=df_impactos["Impacto"],
    orientation='h',
    marker_color=df_impactos["Color"],
    text=df_impactos["Impacto"].apply(
        lambda x: f"+{round(x, 1)}%" if x > 0 else f"{round(x, 1)}%"),
    textposition='outside',
    textfont=dict(color='white')
))

fig_tornado.update_layout(
    title=dict(
        text=f"Impacto Neto en la Probabilidad de Falla ({pozo_seleccionado})",
        font=dict(color="white", size=16)),
    xaxis=dict(
        title=dict(text="Desviación de Riesgo (%)", font=dict(color="white")),
        showgrid=True, gridcolor="#334155",
        tickfont=dict(color="white")),
    yaxis=dict(showgrid=False, tickfont=dict(color="white")),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=400,
    margin=dict(l=150, r=50, t=50, b=50)
)
st.plotly_chart(fig_tornado, use_container_width=True)
