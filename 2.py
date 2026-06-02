import google.genai as genai
import streamlit as st
import datetime
from fpdf import FPDF
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
import streamlit.components.v1 as components

import dss
import app1

# ================= CONFIGURACIÓN =================
st.set_page_config(page_title="Taller de Motos Casa Tuning", layout="wide")

# ================= SIDEBAR (permanente) =================
with st.sidebar:
    st.image("logotuning_120526.png", width=160)
    st.markdown("**Taller de Motos Casa Tuning**")
    st.markdown("📍 Del INEP una cuadra al sur")
    st.markdown("📞 8832-9893")
    if 'ventas_diarias' in st.session_state and st.session_state.ventas_diarias:
        df = pd.DataFrame(st.session_state.ventas_diarias)
        st.metric("Ventas hoy", f"${df['total'].sum():.2f}")

# ================= ENCABEZADO PRINCIPAL =================
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("logotuning_120526.png", width=160)
with col_title:
    st.markdown("<h1 style='color:white;'> Taller de Motos Casa Tuning</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #444;'>", unsafe_allow_html=True)

st.markdown("""
<style>
/* Contenedor general de las pestañas para quitar líneas innecesarias */
div[data-testid="stTabs"] {
    border: none !important;
}

/* Estilo para la lista/barra que aloja las pestañas */
div[data-testid="stTabs"] [role="tablist"] {
    background-color: #2b2b2b !important;
    padding: 6px !important;
    border-radius: 12px !important;
    border-bottom: none !important;
    gap: 10px !important;
}

/* Estilo base para cada pestaña individual (Inactivas) */
div[data-testid="stTabs"] button {
    background-color: transparent !important;
    color: #b3b3b3 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 10px 20px !important;
    transition: all 0.3s ease !important;
}

/* Efecto al pasar el mouse por encima (Hover) */
div[data-testid="stTabs"] button:hover {
    background-color: rgba(0, 191, 255, 0.1) !important;
    color: #00BFFF !important;
}

/* Estilo para la pestaña seleccionada (Activa) */
div[data-testid="stTabs"] button[aria-selected="true"] {
    background-color: #00BFFF !important;
    color: #1e1e1e !important; /* Texto oscuro para que resalte y sea legible */
    box-shadow: 0px 4px 12px rgba(0, 191, 255, 0.4) !important;
}

/* Ocultar la línea roja/azul por defecto que Streamlit dibuja debajo de la pestaña activa */
div[data-testid="stTabs"] [data-baseweb="tab-highlight-bar"] {
    background-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Facturación", " Análisis de Sentimiento", " DSS"])
# ================= FACTURACIÓN =================
with tab1:
    st.title(" Sistema de Facturación")
    st.session_state.active_tab = "facturacion"

    # ================= CONFIGURACIÓN =================
    def conectar_gemini():
        return genai.Client(api_key="AQ.Ab8RN6LCFZCFfiqSkB3aqfWjITfV2tmz-To6Y6HAVg0plsz2GQ")

    client = conectar_gemini()

    #================= SESSION STATE =================
    if 'ventas_diarias' not in st.session_state:
        st.session_state.ventas_diarias = []
    if 'items_factura' not in st.session_state:
        st.session_state.items_factura = []  
    if 'descargar_pdf' not in st.session_state:
        st.session_state.descargar_pdf = None
    if 'trigger_download' not in st.session_state:
        st.session_state.trigger_download = False
    if 'form_reset' not in st.session_state:
        st.session_state.form_reset = 0
    if 'item_reset' not in st.session_state:
        st.session_state.item_reset = 0

    # ================= ESTILOS =================
    st.markdown("""
    <style>
    .stApp { background-color: #1e1e1e !important; }
    [data-testid="stSidebar"] { background-color: #2b2b2b !important; }
    h1,h2,h3,label,.stMarkdown { color: white !important; }
    .stButton button {
        background-color: #00BFFF !important;
        color: white !important;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }
    .stButton button:hover { background-color: #009acd !important; }
    .stTextInput input, .stTextArea textarea, .stNumberInput input, .stSelectbox select {
        background-color: #333 !important;
        color: white !important;
        border: 1px solid #00BFFF !important;
    }
    [data-testid="stMetricValue"] { color: #FFD700 !important; }
    </style>
    """, unsafe_allow_html=True)

    # ================= FUNCIONES =================
    def generar_factura(prompt):
        modelos = [
            "gemini-2.5-flash",
            "gemini-flash-latest",
            "gemini-3.5-flash",
            "gemini-2.5-pro"
        ]
        for modelo in modelos:
            try:
                response = client.models.generate_content(
                    model=modelo,
                    contents=prompt
                )
                return response.text
            except Exception as e:
                st.warning(f" Modelo {modelo} no disponible ({e}). Probando otro...")
        st.error(" Ningún modelo disponible en este momento. Intenta más tarde.")
        return None

    # CORRECCIÓN AQUÍ: Modificado para que sea compatible con fpdf2 moderno
    def crear_pdf(factura_texto):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Taller de Motos Casa Tuning", ln=True, align="C")
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 8, "Del INEP una cuadra al sur - Matagalpa", ln=True, align="C")
        pdf.cell(0, 8, "Whatsapp: 8832-9893", ln=True, align="C")
        pdf.ln(8)
        pdf.set_font("Arial", size=9)
        texto = factura_texto.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 5, texto)
        
        # Guardar en memoria correctamente en versiones nuevas de fpdf
        pdf_output = io.BytesIO(pdf.output())
        pdf_output.seek(0)
        return pdf_output

    def descarga_automatica(data, filename):
        b64 = base64.b64encode(data).decode()
        html = f"""
        <html><body>
            <a id="download" href="data:application/pdf;base64,{b64}" download="{filename}"></a>
            <script>document.getElementById("download").click();</script>
        </body></html>
        """
        components.html(html, height=0)

    # ================= LAYOUT =================
    col_left, col_right = st.columns([2, 1])

    # ================= PANEL DERECHO =================
    with col_right:
        st.markdown("### Reporte")
        if st.session_state.ventas_diarias:
            df = pd.DataFrame(st.session_state.ventas_diarias)
            servicios = df['producto'].value_counts()
            colores = ['#00BFFF','#FF6B35','#FFD700','#8A2BE2','#00FA9A','#FF1493']
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.pie(servicios.values, labels=servicios.index, autopct='%1.1f%%', colors=colores[:len(servicios)])
            ax.set_facecolor('#1e1e1e')
            fig.patch.set_facecolor('#2c2c2c')
            st.pyplot(fig)

    # ================= FORMULARIO FACTURACIÓN =================
    with col_left:
        st.markdown("###  Datos del Cliente")
        
        cliente_bloqueado = len(st.session_state.items_factura) > 0
        
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            cliente = st.text_input("Cliente", key=f"cliente_{st.session_state.form_reset}", disabled=cliente_bloqueado)
        with col_c2:
            moto = st.text_input("Moto", key=f"moto_{st.session_state.form_reset}", disabled=cliente_bloqueado)
        with col_c3:
            placa = st.text_input("Placa", key=f"placa_{st.session_state.form_reset}", disabled=cliente_bloqueado)
            
        metodo_pago = st.selectbox("Método de Pago", ["Efectivo","Transferencia","Tarjeta"], key=f"metodo_{st.session_state.form_reset}", disabled=cliente_bloqueado)

        st.markdown("---")
        st.markdown("###  Agregar Productos / Servicios")
        
        col1, col2 = st.columns(2)
        with col1:
            producto = st.selectbox("Producto/Servicio", [
                "Reparación Mecánica Mayor","Mantenimiento Rutinario",
                "Reparación Eléctrica","Diagnóstico/Evaluación","Modificación/Tuning"
            ], key=f"item_producto_{st.session_state.item_reset}")
            descripcion = st.text_area("Descripción del servicio", key=f"item_descripcion_{st.session_state.item_reset}")
        with col2:
            cantidad = st.number_input("Cantidad", min_value=1, value=1, key=f"item_cantidad_{st.session_state.item_reset}")
            precio = st.number_input("Precio Unitario", min_value=0.0, value=0.0, key=f"item_precio_{st.session_state.item_reset}")
        
        if st.button(" Agregar Item a la Tabla", use_container_width=True):
            if not cliente:
                st.error(" Primero debes ingresar el nombre del cliente antes de añadir productos.")
            elif precio > 0:
                subtotal = cantidad * precio
                nuevo_item = {
                    "Producto/Servicio": producto,
                    "Descripción": descripcion,
                    "Cantidad": cantidad,
                    "Precio Unitario": precio,
                    "Subtotal": subtotal
                }
                st.session_state.items_factura.append(nuevo_item)
                st.toast("🛒 Artículo añadido temporalmente")
                
                st.session_state.item_reset += 1
                st.rerun()
            else:
                st.error("⚠️ El precio unitario debe ser mayor a 0")

        # 3. TABLA INTERACTIVA DE ARTÍCULOS (MEJORADA)
        st.markdown("### 📋 Items Añadidos")
        st.caption("💡 Para eliminar una fila: selecciónala haciendo clic en el extremo izquierdo y presiona la tecla 'Supr' (Delete) de tu teclado.")
        
        if st.session_state.items_factura:
            df_items = pd.DataFrame(st.session_state.items_factura)
            
            # Al usar num_rows="dynamic", Streamlit permite al usuario borrar y añadir filas interactivamente
            df_editado = st.data_editor(df_items, use_container_width=True, num_rows="dynamic")
            
            # Comparamos si cambió el número de registros (eliminación) o sus valores para actualizar el estado real
            if not df_editado.equals(df_items):
                st.session_state.items_factura = df_editado.to_dict(orient="records")
                st.rerun() # Recargamos para refrescar el cálculo del total inmediatamente
            
            total_factura = df_editado["Subtotal"].sum() if not df_editado.empty else 0.0
            st.markdown(f"#### **Total acumulado: ${total_factura:.2f}**")
        else:
            st.info("La tabla está vacía. Agrega items arriba.")

        st.markdown("---")

        # 4. BOTÓN FINALIZAR FACTURACIÓN
        if st.button("✅ Finalizar Facturación", use_container_width=True):
            if not cliente:
                st.error("❌ Por favor escribe el nombre del cliente.")
            elif not st.session_state.items_factura:
                st.error("❌ No puedes facturar si la tabla está vacía.")
            else:
                fecha = datetime.datetime.now().strftime("%d/%m/%Y")
                hora = datetime.datetime.now().strftime("%H:%M")
                
                items_texto = ""
                for idx, item in enumerate(st.session_state.items_factura, 1):
                    items_texto += f"""
                    Item {idx}:
                    - Servicio: {item['Producto/Servicio']}
                    - Descripción: {item['Descripción']}
                    - Cantidad: {item['Cantidad']}
                    - Precio Unitario: ${item['Precio Unitario']:.2f}
                    - Subtotal: ${item['Subtotal']:.2f}
                    -------------------------------"""

                total_general = sum(item['Subtotal'] for item in st.session_state.items_factura)

                prompt = f"""
                Genera una factura profesional sin tablas para el Taller de Motos Casa Tuning Matagalpa, Del INEP una cuadra al sur. Numero de Whatsapp: 8832-9893. No saludes ni des despedidas, solo genera la factura con el siguiente formato, mejora la presentación de la factura y pon el logo al inicio al lado del nombre del taller:

                Datos del Cliente:
                Cliente: {cliente}
                Moto: {moto}
                Placa: {placa}
                Fecha: {fecha}
                Hora: {hora}
                Método de pago: {metodo_pago}
                logo: logotuning_120526.png

                Detalle de Servicios Prestados:
                {items_texto}

                TOTAL GENERAL A PAGAR: ${total_general:.2f}

                Agrega al final:
                "Todo se reduce a no rendirse nunca"
                """
                
                with st.spinner("Generando factura con Gemini..."):
                    factura = generar_factura(prompt)
                    if factura:
                        pdf = crear_pdf(factura)
                        filename = f"Factura_{cliente}_{fecha}.pdf"
                        st.session_state.descargar_pdf = {"data": pdf.getvalue(),"filename": filename}
                        
                        for item in st.session_state.items_factura:
                            st.session_state.ventas_diarias.append({
                                "producto": item['Producto/Servicio'],
                                "cliente": cliente,
                                "total": item['Subtotal']
                            })
                        
                        st.session_state.items_factura = []
                        st.session_state.form_reset += 1
                        st.session_state.item_reset += 1  
                        st.session_state.trigger_download = True
                        st.rerun()

        if st.session_state.trigger_download:
            st.success("✅ Factura generada con éxito. Los campos han sido liberados y limpiados.")
            descarga_automatica(st.session_state.descargar_pdf["data"], st.session_state.descargar_pdf["filename"])
            st.download_button("📥 Descargar manualmente", data=st.session_state.descargar_pdf["data"],
                            file_name=st.session_state.descargar_pdf["filename"], mime="application/pdf")
            st.session_state.trigger_download = False
            
    # ================= ANÁLISIS DE SENTIMIENTO =================
    with tab2:
        st.title("Análisis de Sentimiento")
        app1.run_app()

    # ================= DSS =================
    with tab3:
        st.title("Modelo Predictivo Y DSS de Fidelización")
        dss.run_app()