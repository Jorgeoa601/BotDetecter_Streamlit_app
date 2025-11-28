# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Importaciones de Módulos ---
from src.features import procesar_datos
from src.model import entrenar_y_predecir

# ================= CONFIGURACIÓN DE PÁGINA =================
st.set_page_config(
    page_title="BotBuster: Detector de Anomalías",
    page_icon="🤖",
    layout="wide"
)

# ================= ESTILOS CSS PERSONALIZADOS =================
st.markdown("""
<style>
    .main-header {font-size: 36px; font-weight: bold; color: #1E88E5; text-align: center;}
    .sub-header {font-size: 18px; color: #666; text-align: center; margin-bottom: 20px;}
    .card {background-color: #f9f9f9; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .metric-box {text-align: center; padding: 10px; background: #e3f2fd; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)


# ================= INTERFAZ PRINCIPAL =================
st.markdown('<div class="main-header">🤖 Detector de Bots con BETO & Isolation Forest</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Análisis semántico y conductual para detección de anomalías</div>', unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("📂 Configuración")
opcion = st.sidebar.radio("Datos de entrada:", ["Dataset de Ejemplo", "Subir CSV"])

# Slider para el umbral de riesgo (Rango ajustado de 0.0 a 1.0)
umbral_risk_score = st.sidebar.slider(
    "Umbral de Detección de Bots (Risk Score)",
    min_value=0.0,
    max_value=1.0,
    value=0.75, 
    step=0.01,
    format='%.2f'
)
st.sidebar.info(f"Un usuario se clasifica como Bot si su Risk Score es > {umbral_risk_score*100:.0f}%")

df_raw = None

if opcion == "Dataset de Ejemplo":
    # Datos sintéticos más robustos para el ejemplo
    data = {
        'user_id': ['bot1']*5 + ['userA']*3 + ['bot2']*10 + ['userB']*4,
        'text': [
            'Gana dinero rápido! Visita mi bio: http://spam.com', 'Gana dinero rápido! Visita mi bio: http://spam.com', 'Click en mi bio', 'Gana dinero rápido! Visita mi bio: http://spam.com', 'Oferta limitada! @otro_user', # Bot 1 (Repetitivo con ruido)
            'Me gustó mucho el video, gran trabajo 👍', 'Gracias por la info, muy útil.', '¿Podrías hablar de Python? #coding', # User A (Normal)
            'Suscríbete', 'Suscríbete', 'Suscríbete', 'Suscríbete', 'Suscríbete', 'Suscríbete', 'Suscríbete', 'Suscríbete', 'Suscríbete', 'Suscríbete', # Bot 2 (Spam agresivo)
            'Jajaja que risa 😂', 'No estoy de acuerdo con eso.', 'Buen punto, lo consideraré.', 'Saludos desde Chile' # User B (Normal)
        ],
        'timestamp': pd.to_datetime(['2023-01-01 12:00']*5 + ['2023-01-01 14:00']*3 + ['2023-01-01 03:00']*10 + ['2023-01-02 10:00']*4),
        'likes': [0,0,1,0,0, 10,5,2, 0,0,0,0,0,0,0,0,0,0, 12,3,5,8],
        'replies': [0]*5 + [2,1,4] + [0]*10 + [1,0,2,3]
    }
    df_raw = pd.DataFrame(data)
    st.info("ℹ️ Usando dataset de demostración generado.")
else:
    file = st.sidebar.file_uploader("Sube tu CSV (cols: user_id, text, timestamp, likes, replies)", type="csv")
    if file:
        df_raw = pd.read_csv(file)
    else:
        st.warning("Sube un archivo para continuar.")
        st.stop()

# ================= 5. DETALLE DEL PIPELINE DE PROCESAMIENTO (DIDÁCTICO) =================
if df_raw is not None:
    # 1. LLAMADA A INGENIERÍA DE RASGOS (features.py)
    with st.spinner('⚙️ Procesando textos y calculando métricas de comportamiento...'):
        df_users, df_processed_posts = procesar_datos(df_raw)
        
        # Validación de nulos (Crucial para no romper sklearn)
        df_users = df_users.fillna(0)

    st.header("Detalle del Pipeline de Procesamiento")
    st.markdown("Esta sección muestra los datos antes de la detección de anomalías.")

    # --- Tabla 1: Datos Crudos ---
    with st.expander("▶️ 4.1. Visualización de Datos Crudos (Input)"):
        st.subheader("Datos de Entrada Originales")
        st.dataframe(df_raw, use_container_width=True)

    # --- Tabla 2: Métrica de Post Limpio (Detalle por fila) ---
    with st.expander("▶️ 4.2. Preprocesamiento (Textos Limpios)"):
        st.subheader("Texto Después de la Limpieza")
        st.markdown("URLs, menciones, hashtags y emojis han sido eliminados.")
        st.dataframe(df_processed_posts, use_container_width=True, hide_index=True)


    # --- Tabla 3: Ingeniería de Rasgos (Agrupado por usuario) ---
    with st.expander("▶️ 4.3. Ingeniería de Rasgos (Métricas Agrupadas por Usuario)"):
        st.subheader("Variables de Comportamiento y Léxicas")
        st.markdown("Estas variables se usan para entrenar el modelo de detección de anomalías, junto con los *embeddings*.")
        
        # Seleccionamos las columnas clave para mostrar todas las métricas calculadas
        cols_to_display = [
            'user_id', 'num_posts', 
            'intervalo_medio', 
            'frecuencia_diaria', 'nocturnidad', 
            'avg_likes', 'avg_replies', 
            'ttr', 'tasa_repeticion', 'longitud_promedio'
        ]
        
        df_metrics_display = df_users[cols_to_display].copy()
        
        # IMPORTANTE: Eliminamos el formateo a STRING de nocturnidad y tasa_repeticion
        # para que Streamlit pueda usar column_config.NumberColumn
        df_metrics_display['intervalo_medio'] = df_metrics_display['intervalo_medio'].map(lambda x: f'{x:.2f}')
        df_metrics_display['frecuencia_diaria'] = df_metrics_display['frecuencia_diaria'].map(lambda x: f'{x:.2f}')
        df_metrics_display['nocturnidad'] = df_metrics_display['nocturnidad'].map(lambda x: f'{x:.2%}')
        df_metrics_display['tasa_repeticion'] = df_metrics_display['tasa_repeticion'].map(lambda x: f'{x:.2%}')
        # Los demás valores (nocturnidad, tasa_repeticion) se dejan como float (0 a 1)
        
        st.dataframe(
            df_metrics_display, 
            use_container_width=True,
            column_config={
                "intervalo_medio": st.column_config.NumberColumn("Intervalo Medio (min)", format="%.2f min"),
                "frecuencia_diaria": st.column_config.NumberColumn("Posts/Día (Aprox)", format="%.2f"),
                "nocturnidad": "Actividad Nocturna",
                "ttr": "Diversidad Léxica",
                "tasa_repeticion": "Tasa de Repetición",
                "longitud_promedio": "Longitud Promedio",
                "avg_likes": "Likes Promedio",
                "avg_replies": "Respuestas Promedio"
            },
            hide_index=True
        )
    
    st.divider()
    
    # 2. LLAMADA AL MODELO (model.py)
    df_users, features_conducta = entrenar_y_predecir(df_users, umbral_risk_score)

    st.header("Resultados del Modelo de Detección de Anomalías")

    # ================= RESULTADOS GRÁFICOS (MÚLTIPLES GRÁFICOS) =================
    
    st.subheader("📊 Comparación Individualizada de Métricas: Bots vs. Usuarios Normales")
    st.markdown("Promedio de cada métrica de comportamiento y léxica, agrupado por la clasificación final. **(Escalas ajustadas por métrica)**")
    
    # --- Preparación de datos para el gráfico de comparación ---
    df_comparison = df_users.groupby('es_bot')[features_conducta].mean().T.reset_index()
    df_comparison.columns = ['Metrica', 'Usuario Normal', 'Bot Detectado']
    
    metric_map = {
        'num_posts': 'Posts Totales',
        'intervalo_medio': 'Intervalo Medio (min)',
        'frecuencia_diaria': 'Frecuencia Diaria',
        'nocturnidad': 'Actividad Nocturna (00-06h)',
        'ttr': 'Diversidad Léxica (TTR)',
        'tasa_repeticion': 'Tasa de Repetición',
        'avg_likes': 'Likes Promedio',
        'avg_replies': 'Respuestas Promedio',
        'longitud_promedio': 'Longitud Promedio'
    }
    
    # --- Creación de Gráficos Individuales ---
    col1_graficos, col2_top = st.columns([2, 1])
    
    with col1_graficos:
        cols = st.columns(2)
        col_index = 0
        
        for original_metric, display_name in metric_map.items():
            df_plot = df_comparison[df_comparison['Metrica'] == original_metric].copy()
            df_long_single = pd.melt(df_plot, id_vars='Metrica', var_name='Clasificación', value_name='Valor Promedio')
            
            with cols[col_index % 2]:
                fig_bar_single = px.bar(
                    df_long_single, 
                    x='Clasificación', 
                    y='Valor Promedio', 
                    color='Clasificación', 
                    height=300,
                    title=f'**{display_name}**',
                    color_discrete_map={
                        'Bot Detectado': 'salmon', 
                        'Usuario Normal': 'lightseagreen'
                    },
                    labels={'Valor Promedio': 'Valor Promedio', 'Clasificación': 'Tipo de Usuario'}
                )
                fig_bar_single.update_layout(
                    xaxis_title=None,
                    title_x=0.5,
                    legend_title_text='Tipo de Usuario',
                    showlegend=True if col_index == 0 else False
                )
                st.plotly_chart(fig_bar_single, use_container_width=True)
            col_index += 1

    with col2_top:
        st.subheader("🚨 Top Sospechosos")
        sospechosos = df_users.sort_values('risk_score', ascending=False).head(10)
        
        st.dataframe(
            sospechosos[['user_id', 'risk_score', 'num_posts']],
            column_config={
                "risk_score": st.column_config.ProgressColumn(
                    "Nivel de Riesgo",
                    help="Probabilidad de que el usuario sea un bot",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                    width="medium",
                ),
                "num_posts": st.column_config.NumberColumn("Posts Totales", format="%d")
            },
            hide_index=True,
            use_container_width=True
        )

    st.divider()
    
    # ================= 6. CLASIFICACIÓN FINAL DE USUARIOS (DETALLADA) =================
    st.header("Clasificación Final de Usuarios")
    
    total_bots = df_users['es_bot'].sum()
    st.metric(label=f"Total de Usuarios Clasificados como Bots (Umbral > {umbral_risk_score*100:.0f}%)", value=total_bots, delta_color="inverse")
    
    st.markdown(f"La siguiente tabla muestra la clasificación final basada en el umbral dinámico de **Risk Score** (> {umbral_risk_score*100:.0f}%) e incluye todas las **métricas de comportamiento y léxicas** usadas en la detección.")
    
    features_conducta_display = [
        'num_posts', 
        'intervalo_medio', 
        'frecuencia_diaria', 
        'avg_likes', 
        'avg_replies', 
        'ttr', 
        'longitud_promedio'
    ]
    
    cols_to_show_final = ['user_id', 'risk_score', 'es_bot'] + features_conducta_display
    df_final_detailed = df_users[cols_to_show_final].copy()
    
    df_final_detailed['Clasificación'] = df_final_detailed['es_bot'].map({
        True: '🤖 Bot Detectado', 
        False: '👤 Usuario Normal'
    })
    
    column_config_map = {
        "user_id": "ID de Usuario",
        "Clasificación": "Clasificación Final",
        "risk_score": st.column_config.ProgressColumn(
            "Nivel de Riesgo",
            help="Probabilidad de que el usuario sea un bot (0-100%)",
            format="%.2f",
            min_value=0,
            max_value=1,
            width="medium"
        ),
        "intervalo_medio": st.column_config.NumberColumn("Intervalo Medio (min)", format="%.2f min"),
        "frecuencia_diaria": st.column_config.NumberColumn("Posts/Día (Aprox)", format="%.2f"),
        "ttr": st.column_config.NumberColumn("Diversidad Léxica (TTR)", format="%.2f"),
        "longitud_promedio": st.column_config.NumberColumn("Longitud Promedio", format="%.1f"),
        "avg_likes": st.column_config.NumberColumn("Likes Promedio", format="%.1f"),
        "avg_replies": st.column_config.NumberColumn("Respuestas Promedio", format="%.1f"),
        "num_posts": st.column_config.NumberColumn("Posts Totales", format="%d"),
        "es_bot": None
    }
    
    display_order = ['user_id', 'Clasificación', 'risk_score'] + features_conducta_display
    
    st.dataframe(
        df_final_detailed[display_order].sort_values(by='risk_score', ascending=False),
        column_config=column_config_map,
        hide_index=True,
        use_container_width=True
    )
    
    st.divider()

    # ================= 7. ANÁLISIS INDIVIDUAL (SOLO MÉTRICAS) =================
    st.subheader("Análisis Detallado por Usuario")
    usuario_selec = st.selectbox("Seleccionar Usuario para auditar:", df_users['user_id'].unique())
    
    if usuario_selec:
        user_data = df_users[df_users['user_id'] == usuario_selec].iloc[0]
        
        # --- FILA 1: Risk Score, Posts Totales y Rasgos de Comportamiento ---
        st.markdown('**Comportamiento y Métricas Generales**')
        col_r1_1, col_r1_2, col_r1_3, col_r1_4, col_r1_5 = st.columns(5)
        
        clasificacion = '🤖 BOT' if user_data['es_bot'] else '👤 Usuario'
        col_r1_1.metric(f"Risk Score ({clasificacion})", f"{user_data['risk_score']:.2%}", delta_color="inverse")
        col_r1_2.metric("Posts Totales", user_data['num_posts'])
        col_r1_3.metric("Intervalo Medio", f"{user_data['intervalo_medio']:.2f} min")
        col_r1_4.metric("Frecuencia Diaria", f"{user_data['frecuencia_diaria']:.2f}")
        col_r1_5.metric("Actividad Nocturna", f"{user_data['nocturnidad']:.2%}", delta_color="inverse") 
        
        st.markdown('**Rasgos Léxicos y de Interacción**')
        # --- FILA 2: Rasgos Léxicos y de Interacción ---
        col_r2_1, col_r2_2, col_r2_3, col_r2_4, col_r2_5 = st.columns(5)
        
        col_r2_1.metric("Diversidad Léxica (TTR)", f"{user_data['ttr']:.2f}") 
        col_r2_2.metric("Tasa de Repetición", f"{user_data['tasa_repeticion']:.2%}", delta_color="inverse")
        col_r2_3.metric("Longitud Promedio", f"{user_data['longitud_promedio']:.2f}")
        col_r2_4.metric("Likes Promedio", f"{user_data['avg_likes']:.2f}")
        col_r2_5.metric("Respuestas Promedio", f"{user_data['avg_replies']:.2f}")

        # Muestra de textos
        with st.expander("Ver contenido textual del usuario"):
            st.write(user_data['textos_raw'])
            
    # ================= 8. DESCARGA DE RESULTADOS FINALES =================
    st.divider()
    st.header("Descarga de Resultados")
    st.markdown("Descargue el archivo CSV con la clasificación final de todos los usuarios, incluyendo su Nivel de Riesgo (Risk Score).")

    # 1. Preparar el DataFrame para la descarga
    df_download = df_users[['user_id', 'risk_score', 'es_bot']].copy()

    df_download['Clasificacion_Final'] = df_download['es_bot'].map({
        True: 'BOT DETECTADO', 
        False: 'USUARIO NORMAL'
    })
    
    df_download.rename(columns={
        'user_id': 'ID_Usuario',
        'risk_score': 'Nivel_Riesgo',
    }, inplace=True)
    
    df_export = df_download[['ID_Usuario', 'Nivel_Riesgo', 'Clasificacion_Final']]

    # 2. Convertir el DataFrame a CSV
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(df_export)
    
    # 3. Mostrar el botón de descarga
    st.download_button(
        label="Descargar Clasificación de Usuarios (CSV)",
        data=csv_data,
        file_name='BotBuster_Resultados_Clasificacion.csv',
        mime='text/csv',
        type="primary"
    )