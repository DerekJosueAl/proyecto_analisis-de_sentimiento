import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import io
import os 
from dotenv import load_dotenv
import google.genai as genai
import smtplib

# Cargar variables desde .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMAIL_REMITENTE = os.getenv("EMAIL_REMITENTE")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# Conectar Gemini
client = genai.Client(api_key=GEMINI_API_KEY)

# ================= DSS APP =================
def run_app():
    
    # --- 1. Cargar datos ---
    st.sidebar.header("📂 Carga de Datos")
    uploaded_file = st.sidebar.file_uploader("Sube tu archivo dataset.csv", type=["csv"])

    @st.cache_data
    def process_data(df_input):
        df_copy = df_input.copy()
        df_copy["FechaInicio"] = pd.to_datetime(df_copy["FechaInicio"])
        df_copy["FechaFin"] = pd.to_datetime(df_copy["FechaFin"])
        df_copy["DuracionDias"] = (df_copy["FechaFin"] - df_copy["FechaInicio"]).dt.days
        df_copy["AnioInicio"] = df_copy["FechaInicio"].dt.year
        return df_copy

    df = None
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            df = process_data(df_raw)
            st.success("✅ Archivo subido y procesado con éxito.")
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
    else:
        try:
            df_raw = pd.read_csv("reparaciones_10000_dss_rebalanceado.csv")
            df = process_data(df_raw)
            st.info("📌 Usando dataset local predeterminado.")
        except FileNotFoundError:
            st.warning("⚠️ No se encontró el archivo local. Por favor, sube un CSV.")

    # --- 2. Procesamiento ---
    if df is not None:
        columnas_requeridas = ["FechaInicio","FechaFin","Total","cliente","IdReparacion","Marca","Estado","Fase","Modelo","Descripcion"]
        if not all(col in df.columns for col in columnas_requeridas):
            st.error("⚠️ El dataset no contiene todas las columnas necesarias.")
            st.stop()

        anio_min = df["AnioInicio"].min()
        anio_max = df["AnioInicio"].max()
        total_anios = max(1, (anio_max - anio_min) + 1)

        cliente_stats = df.groupby("cliente").agg(
            total_gastado=("Total","sum"),
            num_reparaciones=("IdReparacion","count"),
            promedio_duracion=("DuracionDias","mean"),
            ultima_fecha=("FechaInicio","max")
        ).reset_index()

        cliente_stats["visitas_al_anio"] = cliente_stats["num_reparaciones"] / total_anios
        hoy = pd.Timestamp.now().normalize()
        cliente_stats["dias_ultima_reparacion"] = (hoy - pd.to_datetime(cliente_stats["ultima_fecha"])).dt.days

        # --- 3. Clasificación ---
        gasto_q25 = cliente_stats["total_gastado"].quantile(0.25)
        gasto_q75 = cliente_stats["total_gastado"].quantile(0.75)

        def clasificar_cliente(row):
            if row["num_reparaciones"] >= 26 and row["total_gastado"] >= gasto_q75:
                return "VIP"
            elif 11 <= row["num_reparaciones"] <= 25:
                return "Normal"
            else:
                return "En riesgo"

        cliente_stats["categoria"] = cliente_stats.apply(clasificar_cliente, axis=1)

        # --- 4. Recomendaciones ---
        def recomendar_fidelizacion(categoria, dias_inactivo, gasto, visitas):
            if categoria == "VIP":
                return "Cliente VIP. Mantenimiento gratuito, prioridad y membresía exclusiva."
            elif categoria == "Normal":
                return "Cliente Normal. Programa de puntos, descuentos y comunicación personalizada."
            else:
                if dias_inactivo > 90:
                    return "En riesgo. Inactivo >90 días. Ofrecer 20% descuento o llamada personalizada."
                elif visitas <= 5:
                    return "Baja frecuencia. Cupón de lavado gratis o chequeo rápido."
                elif gasto < gasto_q25:
                    return "Bajo gasto. Promocionar servicios básicos o diagnósticos gratuitos."
                else:
                    return "En riesgo. Encuesta de satisfacción + cupón 20%."

        cliente_stats["recomendacion"] = cliente_stats.apply(
            lambda row: recomendar_fidelizacion(row["categoria"], row["dias_ultima_reparacion"], row["total_gastado"], row["visitas_al_anio"]),
            axis=1
        )

        # --- 5. KPIs ---
        st.subheader("📊 KPIs Globales")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: st.metric("Clientes activos", len(cliente_stats))
        with col2: st.metric("VIP", len(cliente_stats[cliente_stats["categoria"]=="VIP"]))
        with col3: st.metric("Normales", len(cliente_stats[cliente_stats["categoria"]=="Normal"]))
        with col4: st.metric("En riesgo", len(cliente_stats[cliente_stats["categoria"]=="En riesgo"]))
        with col5: st.metric("Ingreso total", f"${df['Total'].sum():,.0f}")

        # --- 6. Gráficos ---
        st.plotly_chart(
            px.pie(cliente_stats, names="categoria", hole=0.3),
            use_container_width=True,
            key="grafico_categoria"
        )

        # --- 7. Tabla con recomendaciones ---
        st.subheader("Clientes y Recomendaciones")
        columnas_mostrar = ["cliente","total_gastado","num_reparaciones","visitas_al_anio","dias_ultima_reparacion","categoria","recomendacion"]
        st.dataframe(cliente_stats[columnas_mostrar], use_container_width=True, height=600)

        # --- 8. Modelo Predictivo ---
        st.markdown("---")
        st.subheader("🔮 Modelo Predictivo")

        df_train = pd.read_csv("reparaciones_10000_dss_rebalanceado.csv")
        df_train = process_data(df_train)

        desc_por_cliente = df_train.groupby("cliente")["Descripcion"].apply(lambda x: " ".join(x.astype(str))).reset_index()
        cliente_train = df_train.groupby("cliente").agg(
            total_gastado=("Total","sum"),
            num_reparaciones=("IdReparacion","count"),
            promedio_duracion=("DuracionDias","mean"),
            ultima_fecha=("FechaInicio","max")
        ).reset_index()
        cliente_train = cliente_train.merge(desc_por_cliente, on="cliente", how="left")

        hoy = pd.Timestamp.now().normalize()
        cliente_train["dias_ultima_reparacion"] = (hoy - pd.to_datetime(cliente_train["ultima_fecha"])).dt.days

        gasto_q25_train = cliente_train["total_gastado"].quantile(0.25)
        gasto_q75_train = cliente_train["total_gastado"].quantile(0.75)

        cliente_train["categoria"] = cliente_train.apply(clasificar_cliente, axis=1)

        vectorizer = TfidfVectorizer(max_features=200)
        X_text = vectorizer.fit_transform(cliente_train["Descripcion"].astype(str))
        X_num = cliente_train[["total_gastado","num_reparaciones","promedio_duracion","dias_ultima_reparacion"]]
        X = hstack([X_num.values, X_text])
        y = cliente_train["categoria"]

        @st.cache_resource
        def entrenar_modelo(_X, y):
            X_train, X_test, y_train, y_test = train_test_split(_X, y, test_size=0.2, random_state=42)
            modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
            modelo_rf.fit(X_train, y_train)
            return modelo_rf

        modelo_rf = entrenar_modelo(X, y)

        importances = modelo_rf.feature_importances_[:X_num.shape[1]]
        feat_names = ["total_gastado","num_reparaciones","promedio_duracion","dias_ultima_reparacion"]
        st.plotly_chart(px.bar(x=feat_names, y=importances, title="Importancia de variables"),
                        use_container_width=True, key="grafico_importancia")

        # --- 9. Descargar reporte Excel ---
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer) as writer:
            cliente_stats.to_excel(writer, sheet_name="Reporte")
        st.download_button("📥 Descargar Reporte Excel", buffer, "reporte.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        
        # --- 10. Descargar reporte PDF ---
        def generar_pdf(stats):
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Reporte de Clientes - Taller Casa Tuning", ln=True, align="C")
            pdf.ln(10)

            for _, row in stats.iterrows():
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, f"Cliente: {row['cliente']} ({row['categoria']})", ln=True)
                pdf.set_font("Arial", "", 11)
                pdf.multi_cell(0, 8, f"Gasto total: ${row['total_gastado']:.2f}\n"
                                     f"Número de reparaciones: {row['num_reparaciones']}\n"
                                     f"Visitas al año: {row['visitas_al_anio']:.2f}\n"
                                     f"Días desde última reparación: {row['dias_ultima_reparacion']}\n"
                                     f"Recomendación: {row['recomendacion']}")
                pdf.ln(5)

            pdf_output = io.BytesIO()
            pdf.output(pdf_output)
            pdf_output.seek(0)
            return pdf_output

        pdf_output = generar_pdf(cliente_stats)
        st.download_button("📥 Descargar Reporte PDF", pdf_output, "reporte.pdf", "application/pdf", use_container_width=True, key="pdf_download")
        
        # --- 11. Clasificación manual (Con Probabilidades y Diagnóstico Dinámico) ---
        st.markdown("---")
        st.subheader("📝 Clasificación de un nuevo cliente")

        with st.form("nuevo_cliente_form", clear_on_submit=True):
            col1, col2 = st.columns(2)

            with col1:
                gasto_input = st.number_input("💰 Total gastado", min_value=0.0, step=10.0)
                reparaciones_input = st.number_input("🔧 Número de reparaciones", min_value=0, step=1)

            with col2:
                duracion_input = st.number_input("⏱️ Duración promedio (días)", min_value=0.0, step=1.0)
                dias_inactivo_input = st.number_input("📅 Días desde última reparación", min_value=0, step=1)

            descripcion_input = st.text_area("📝 Descripción de servicios")

            submitted = st.form_submit_button("Clasificar cliente")

            if submitted:
                if descripcion_input.strip() == "":
                    st.warning("⚠️ Por favor, ingresa una descripción de los servicios.")
                else:
                    # 1. Preparar datos y vectorizar texto
                    desc_vector = vectorizer.transform([descripcion_input])
                    num_vector = [[gasto_input, reparaciones_input, duracion_input, dias_inactivo_input]]
                    X_new = hstack([num_vector, desc_vector])

                    # Predicciones probabilísticas del Random Forest
                    pred_categoria = modelo_rf.predict(X_new)[0]
                    pred_proba = modelo_rf.predict_proba(X_new)[0]
                    clases = modelo_rf.classes_

                    # 2. Clasificación paralela por Reglas de Negocio fijas
                    fila_simulada = {
                        "num_reparaciones": reparaciones_input,
                        "total_gastado": gasto_input,
                        "promedio_duracion": duracion_input,
                        "dias_ultima_reparacion": dias_inactivo_input
                    }
                    regla_categoria = clasificar_cliente(fila_simulada)

                    # 3. Mostrar Outputs Principales
                    st.success(f"🔮 **Predicción del Modelo:** {pred_categoria}")
                    st.info(f"📋 **Según Reglas de Negocio:** {regla_categoria}")

                    # 4. Renderizado visual de Probabilidades numéricas
                    st.write("### 📊 Probabilidades de Clasificación")
                    prob_dict = dict(zip(clases, pred_proba))
                    
                    for clase, prob in prob_dict.items():
                        porcentaje = prob * 100
                        st.write(f"**{clase}:** {porcentaje:.1f}%")
                        st.progress(float(prob))

                    # 5. Desglose analítico del riesgo ("¿Por qué?")
                    st.write("### 🔍 Diagnóstico analítico de alertas")
                    
                    motivos = []
                    if dias_inactivo_input > 90:
                        motivos.append(f"🔴 **Inactividad crítica:** {dias_inactivo_input} días sin visitas (Alerta activa a partir de los 90 días).")
                    elif dias_inactivo_input > 60:
                        motivos.append(f"🟡 **Inactividad moderada:** El cliente acumula {dias_inactivo_input} días ausente.")
                        
                    if reparaciones_input < 11:
                        motivos.append(f"🔴 **Frecuencia baja:** Registra únicamente {reparaciones_input} órdenes de servicio (Se requieren de 11 a 25 para ser 'Normal').")
                        
                    if gasto_input < gasto_q25:
                        motivos.append(f"💵 **Bajo volumen de compra:** Gasto acumulado de ${gasto_input:,.2f}  se ubica por debajo del percentil mínimo del negocio  (${gasto_q25:,.2f}).")
                    
                    # Despliegue de justificaciones condicionales
                    if pred_categoria == "En riesgo" or regla_categoria == "En riesgo":
                        if motivos:
                            for motivo in motivos:
                                st.write(motivo)
                        else:
                            st.write("⚠️ El riesgo de abandono se deriva primordialmente del análisis semántico efectuado sobre el historial de descripciones textuales de sus servicios.")
                    else:
                        st.write("✅ Los parámetros operativos (monto transaccionado y recurrencia de visitas) sitúan al cliente en una zona saludable y fuera de riesgo por el momento.")
    else:
        st.info("📌 Por favor, sube un archivo CSV para comenzar el análisis.")

# Ejecución del aplicativo
if __name__ == "__main__":
    run_app()