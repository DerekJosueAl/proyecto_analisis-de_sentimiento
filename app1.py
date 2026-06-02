# app.py - Aplicación principal de Streamlit con IA Agéntica y Envío de Correo Real
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import google.genai as genai
from utils.preprocess import (
    clean_text, extract_rating, rating_to_sentiment, 
    analyze_problems, PROBLEM_KEYWORDS
)

# ================= CONFIGURACIÓN DE CREDENCIALES VIA SECRETS =================
# Inicializamos el cliente de Gemini de forma global una sola vez de manera segura
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"❌ Error al cargar GEMINI_API_KEY desde st.secrets: {e}")

EMAIL_REMITENTE = st.secrets["EMAIL_REMITENTE"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]

def enviar_correo_real(destinatario, asunto, cuerpo_mensaje):
    """Función para enviar correo por Gmail usando Secrets"""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_REMITENTE
        msg['To'] = destinatario
        msg['Subject'] = asunto
        msg.attach(MIMEText(cuerpo_mensaje, 'plain'))

        # Conexión segura con el servidor SMTP de Gmail (Puerto 587)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_REMITENTE, EMAIL_PASSWORD)
        server.sendmail(EMAIL_REMITENTE, destinatario, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"❌ Error técnico al enviar el correo: {e}")
        return False

def run_app():
    st.set_page_config(
        page_title="Análisis de Sentimiento - Opiniones de Clientes",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("📊 Dashboard de Análisis de Sentimiento")
    st.markdown("---")

    # ================= FUNCIONES AUXILIARES =================
    @st.cache_data
    def load_data(path="casa_tuning_reviews_7000.csv"):
        """Cargar dataset local"""
        if not os.path.exists(path):
            st.error(f"No se encontró el archivo '{path}'")
            return None
        try:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
            return df
        except Exception as e:
            st.error(f"Error al cargar: {e}")
            return None

    def process_data(df):
        """Procesar opiniones de clientes"""
        if df is None: return None
        data = df.copy()
        data = data.dropna(subset=["Review Text"])
        data["Rating_Num"] = data["Rating"].apply(extract_rating)
        data = data.dropna(subset=["Rating_Num"])
        data["Sentiment"] = data["Rating_Num"].apply(rating_to_sentiment)
        data["Clean_Text"] = data["Review Text"].apply(clean_text)
        data = data[data["Clean_Text"] != ""]
        problems_list = data["Review Text"].apply(lambda x: analyze_problems(str(x), PROBLEM_KEYWORDS))
        for problem in PROBLEM_KEYWORDS.keys():
            data[problem] = problems_list.apply(lambda x: 1 if problem in x else 0)
        return data

    # Estado global para mantener las opiniones y logs del agente en memoria de la sesión
    if 'opiniones_simuladas' not in st.session_state:
        st.session_state.opiniones_simuladas = []

    with st.spinner("Cargando opiniones de clientes..."):
        df_raw = load_data()

    if df_raw is not None:
        with st.spinner("Procesando opiniones..."):
            df = process_data(df_raw)
        
        # ========================================================
        # SECCIÓN: SIMULADOR DE ALERTAS Y FORMULARIO
        # ========================================================
        with st.expander("✍️ Registrar Nueva Opinión de Cliente (Simulador de Alertas para la IA)", expanded=True):
            st.markdown("Ingresa una queja real para ver al Agente Autónomo tomar el control y enviar el correo.")
            
            col_sim1, col_sim2, col_sim3 = st.columns([2, 1, 1])
            with col_sim1:
                sim_nombre = st.text_input("Nombre del Cliente", placeholder="Ej. Juan Pérez")
            with col_sim2:
                sim_rating = st.slider("Clasificación (Estrellas)", min_value=1, max_value=5, value=1)
            with col_sim3:
                sim_email_cliente = st.text_input("Correo del Cliente (Para pruebas)", placeholder="tu_correo_personal@gmail.com")
                
            sim_comentario = st.text_area("Escribe la queja o comentario del cliente...", 
                                          placeholder="Ej. El tuning quedó mal armado, la moto vibra demasiado en alta velocidad y no me quieren dar garantía.")
            sim_sucursal = "Matagalpa"
            
            if st.button("🚨 Enviar Opinión y Activar Flujos Autónomos", use_container_width=True):
                if not sim_nombre or not sim_comentario or not sim_email_cliente:
                    st.error("⚠️ Por favor completa todos los campos (Nombre, Correo y Comentario) para poder ejecutar el agente.")
                else:
                    sentimiento_determinado = 'Negativo' if sim_rating <= 2 else ('Neutro' if sim_rating == 3 else 'Positivo')
                    
                    nueva_fila = {
                        "Reviewer Name": sim_nombre,
                        "Rating": f"{sim_rating} estrellas",
                        "Rating_Num": float(sim_rating),
                        "Review Text": sim_comentario,
                        "Sentiment": sentimiento_determinado,
                        "Country": sim_sucursal,
                        "Clean_Text": clean_text(sim_comentario)
                    }
                    
                    problemas_detectados = analyze_problems(sim_comentario, PROBLEM_KEYWORDS)
                    for problem in PROBLEM_KEYWORDS.keys():
                        nueva_fila[problem] = 1 if problem in problemas_detectados else 0
                    
                    st.session_state.opiniones_simuladas.insert(0, nueva_fila)
                    
                    # DETECCIÓN DE CRISIS: Ejecución de la IA Agéntica
                    if sentimiento_determinado == 'Negativo':
                        with st.spinner("🤖 El Agente Autónomo está analizando la queja y redactando la solución..."):
                            try:
                                prompt_agente = f"""
                                Eres el Agente de Contingencia Automatizado de 'Taller de Motos Casa Tuning'.
                                Tu objetivo es recuperar de inmediato a un cliente insatisfecho enviándole una propuesta de solución.
                                
                                Datos del problema detectado:
                                - Nombre del Cliente: {sim_nombre}
                                - Incidente reportado: "{sim_comentario}"
                                
                                Tareas que debes realizar de manera autónoma:
                                1. Redactar una disculpa corporativa, profesional y empática.
                                2. Ofrecele algo a cambio de que tan gravé se mire el comentario (ejemplo: revisión gratuita, descuento, etc) y hazlo de forma creativa.
                                3. Adjuntar un cupón de descuento exclusivo generado por ti con el formato: TUNING-REC-XXXX (reemplaza XXXX con números aleatorios).
                                4. Al final del correo pon que se guarda el derecho de hacee validó el cupon de descuento o el ofrecimiento, si se demuestra que el cliente es un troll o que no tiene una queja real (ejemplo: solo quiere el descuento sin tener un problema real).
                                Instrucciones de formato estrictas:
                                Debes responder ÚNICAMENTE con el cuerpo del correo que se le enviará al cliente. No agregues introducciones, notas o saludos dirigidos a mí. Empieza directamente con el texto del correo.
                                """
                                
                                # Usamos el cliente global ya configurado correctamente arriba
                                response = client.models.generate_content(
                                    model="gemini-2.5-flash",
                                    contents=prompt_agente
                                )
                                correo_generado = response.text
                                
                                # El Agente ejecuta el envío real
                                asunto_mail = f"Disculpa y Solución Inmediata de Casa Tuning para {sim_nombre}"
                                enviado_ok = enviar_correo_real(sim_email_cliente, asunto_mail, correo_generado)
                                
                                if enviado_ok:
                                    st.session_state.agente_ejecucion = {
                                        "destinatario": sim_email_cliente,
                                        "contenido": correo_generado,
                                        "status": "Enviado con Éxito mediante Servidor SMTP de Gmail ✅"
                                    }
                                else:
                                    st.session_state.agente_ejecucion = {
                                        "destinatario": sim_email_cliente,
                                        "contenido": correo_generado,
                                        "status": "Fallo en el canal de salida SMTP (Verifica tus contraseñas de aplicación de Google) ❌"
                                    }
                                    
                            except Exception as e:
                                st.error(f"Error crítico en el cerebro del agente: {e}")
                    else:
                        if "agente_ejecucion" in st.session_state:
                            del st.session_state.agente_ejecucion
                    
                    # Forzamos el rerun para actualizar la interfaz y pintar las métricas/paneles
                    st.rerun()

        # Unir opiniones añadidas a mano con el lote general
        if st.session_state.opiniones_simuladas:
            df_sim_df = pd.DataFrame(st.session_state.opiniones_simuladas)
            df = pd.concat([df_sim_df, df], ignore_index=True)

        if df is not None and len(df) > 0:
            
            # ============================================
            # SIDEBAR CON FILTROS
            # ============================================
            st.sidebar.title("Filtros")
            st.sidebar.markdown("---")
            sentiment_filter = st.sidebar.multiselect("Sentimiento", options=['Positivo', 'Neutro', 'Negativo'], default=['Positivo', 'Neutro', 'Negativo'])
            rating_filter = st.sidebar.multiselect("Rating (estrellas)", options=sorted(df['Rating_Num'].unique()), default=sorted(df['Rating_Num'].unique()))
            problem_filter = st.sidebar.multiselect("Problemas mencionados", options=list(PROBLEM_KEYWORDS.keys()), default=[])
            
            if 'Country' in df.columns:
                branch_options = ['Todas'] + sorted(df['Country'].dropna().unique().tolist())
                branch_filter = st.sidebar.selectbox("Sucursal / Tienda", options=branch_options, index=0)
            
            st.sidebar.markdown("---")
            num_comments = st.sidebar.slider("Opiniones a mostrar", min_value=10, max_value=500, value=50, step=10)
            
            if st.sidebar.button("Limpiar todo e Historial", use_container_width=True):
                st.cache_data.clear()
                st.session_state.opiniones_simuladas = []
                if "agente_ejecucion" in st.session_state: del st.session_state.agente_ejecucion
                st.rerun()
            
            # Aplicar filtros al DataFrame
            df_filtered = df.copy()
            df_filtered = df_filtered[df_filtered['Sentiment'].isin(sentiment_filter)]
            df_filtered = df_filtered[df_filtered['Rating_Num'].isin(rating_filter)]
            if problem_filter:
                df_filtered = df_filtered[df_filtered[problem_filter].sum(axis=1) > 0]
            if 'Country' in df.columns and branch_filter != 'Todas':
                df_filtered = df_filtered[df_filtered['Country'] == branch_filter]
            
            # ========================================================
            # PANEL DE MONITOREO REAL DEL AGENTE AUTÓNOMO
            # ========================================================
            if 'agente_ejecucion' in st.session_state and 'Negativo' in sentiment_filter:
                st.markdown("### ⚙️ Centro de Operaciones del Agente de IA")
                st.markdown(
                    f"""
                    <div style="background-color: #111b27; border: 2px solid #00BFFF; padding: 20px; border-radius: 10px; margin-bottom: 25px;">
                        <h4 style="color: #00BFFF; margin-top: 0; margin-bottom:10px;">🤖 Acción Autónoma: Protocolo de Recuperación Activado</h4>
                        <p style="color: #ffffff; margin-bottom: 5px;"><b>Canal de Comunicación:</b> Google SMTP (Gmail)</p>
                        <p style="color: #ffffff; margin-bottom: 5px;"><b>Destinatario:</b> {st.session_state.agente_ejecucion['destinatario']}</p>
                        <p style="color: #ffffff; margin-bottom: 15px;"><b>Estado del Envío:</b> <span style="font-weight: bold; color: #2ecc71;">{st.session_state.agente_ejecucion['status']}</span></p>
                        <hr style="border: 0.5px solid #2c3e50; margin-bottom: 15px;">
                        <p style="color: #00BFFF; font-weight:bold; margin-bottom: 5px;">Texto completo enviado al cliente:</p>
                        <div style="background-color: #070d14; padding: 15px; border-radius: 5px; color: #e0e0e0; font-family: sans-serif; white-space: pre-wrap; line-height: 1.5;">{st.session_state.agente_ejecucion['contenido']}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

            # ============================================
            # METRICAS PRINCIPALES
            # ============================================
            total = len(df_filtered)
            positivos = len(df_filtered[df_filtered['Sentiment'] == 'Positivo'])
            neutros = len(df_filtered[df_filtered['Sentiment'] == 'Neutro'])
            negativos = len(df_filtered[df_filtered['Sentiment'] == 'Negativo'])
            avg_rating = df_filtered['Rating_Num'].mean() if total > 0 else 0.0
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1: st.metric("Total Opiniones", f"{total:,}")
            with col2: st.metric("Positivas", f"{positivos:,}", f"{(positivos/total)*100:.1f}%" if total > 0 else "0%")
            with col3: st.metric("Neutras", f"{neutros:,}", f"{(neutros/total)*100:.1f}%" if total > 0 else "0%")
            with col4: st.metric("Negativas", f"{negativos:,}", f"{(negativos/total)*100:.1f}%" if total > 0 else "0%")
            with col5: st.metric("Rating Promedio", f"{avg_rating:.2f}", "estrellas")
            
            st.markdown("---")
            
            # Gráficos de barra
            st.subheader("Distribución de Opiniones")
            if total > 0:
                sentiment_counts = df_filtered['Sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentimiento', 'Cantidad']
                colors_bar = {'Positivo': '#2ecc71', 'Neutro': '#f39c12', 'Negativo': '#e74c3c'}
                fig_bar = px.bar(sentiment_counts, x='Sentimiento', y='Cantidad', color='Sentimiento', color_discrete_map=colors_bar, text='Cantidad')
                fig_bar.update_traces(textposition='outside')
                fig_bar.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            st.markdown("---")
            
            # Gráfico de pastel
            st.subheader("Fallas críticas más mencionadas por clientes molestos")
            df_negativos = df_filtered[df_filtered['Sentiment'] == 'Negativo']
            if len(df_negativos) > 0:
                problem_percentages = {prob: (df_negativos[prob].sum() / len(df_negativos)) * 100 for prob in PROBLEM_KEYWORDS.keys()}
                problem_df = pd.DataFrame({'Problema': list(problem_percentages.keys()), 'Porcentaje': list(problem_percentages.values())}).sort_values('Porcentaje', ascending=False)
                fig_pie = px.pie(problem_df.head(6), values='Porcentaje', names='Problema', hole=0.3)
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No hay opiniones negativas en el filtro actual.")
            
            st.markdown("---")
            
            # Renderizado de la lista histórica de opiniones
            st.subheader("Historial Completo de Opiniones")
            opinion_type = st.radio("", ["Todas", "Positivas", "Neutras", "Negativas"], horizontal=True)
            
            if opinion_type == "Positivas": df_display = df_filtered[df_filtered['Sentiment'] == 'Positivo'].head(num_comments)
            elif opinion_type == "Neutras": df_display = df_filtered[df_filtered['Sentiment'] == 'Neutro'].head(num_comments)
            elif opinion_type == "Negativas": df_display = df_filtered[df_filtered['Sentiment'] == 'Negativo'].head(num_comments)
            else: df_display = df_filtered.head(num_comments)
            
            if len(df_display) > 0:
                for idx, row in df_display.iterrows():
                    st.markdown(f"**[{row['Sentiment'].upper()}]** 👤 {row['Reviewer Name'] if pd.notna(row['Reviewer Name']) else 'Anónimo'} | ⭐ {int(row['Rating_Num'])} estrellas\n\n> {row['Review Text']}\n\n---")
            else:
                st.info("No hay registros disponibles para mostrar.")

    # ============================================
    # CARGAR ARCHIVOS EXTERNOS CSV O XLSX
    # ============================================
    st.markdown("### 📂 Carga de Auditorías Externas")
    uploaded_file = st.file_uploader("Sube un archivo masivo si deseas reescribir la base de datos visualizada", type=["csv", "xlsx"])
    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df_uploaded = process_data(df_uploaded)
        if df_uploaded is not None and len(df_uploaded) > 0:
            st.success("Archivo externo procesado con éxito.")

if __name__ == "__main__":
    run_app()
