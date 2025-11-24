import streamlit as st
import pandas as pd

import re
import unicodedata

from sentence_transformers import SentenceTransformer
import numpy as np

from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title="Detección de Bots en Redes Sociales", layout="wide")

# Título y descripción
st.title("Detección de Bots en Redes Sociales")
st.markdown("""
Esta aplicación permite detectar posibles cuentas automatizadas (bots) en una red social, utilizando técnicas de procesamiento de lenguaje natural (NLP) y modelos de detección de anomalías.
""")

# Cargar archivo CSV o usar dataset de ejemplo
st.sidebar.header("Carga de datos")
opcion = st.sidebar.radio("Selecciona una opción:", ("Subir archivo CSV", "Usar dataset de ejemplo"))

def cargar_ejemplo():
	data = {
		'user_id': ['u1', 'u2', 'u1', 'u3'],
		'text': [
			'Me encanta la inteligencia artificial!',
			'Compra seguidores baratos aquí',
			'Hoy es un gran día para aprender',
			'Sigue mi canal para más sorteos'
		],
		'timestamp': ['2025-11-20', '2025-11-20', '2025-11-21', '2025-11-21'],
		'likes': [10, 0, 5, 1],
		'replies': [2, 0, 1, 0]
	}
	return pd.DataFrame(data)

# --- Preprocesamiento de texto ---
def limpiar_texto(texto):
	if pd.isnull(texto):
		return ""
	# Quitar acentos
	texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8', 'ignore')
	# Convertir a minúsculas
	texto = texto.lower()
	# Quitar caracteres especiales y números
	texto = re.sub(r'[^a-zA-Z\s]', '', texto)
	# Quitar espacios extra
	texto = re.sub(r'\s+', ' ', texto).strip()
	return texto

# --- Agrupamiento por usuario ---
def agrupar_por_usuario(df):
	# Agrupa los textos y suma interacciones por usuario
	agrupado = df.groupby('user_id').agg({
		'text': lambda x: ' '.join(x),
		'likes': 'sum',
		'replies': 'sum',
		'timestamp': 'count'  # número de publicaciones
	}).rename(columns={'timestamp': 'num_posts'})
	agrupado = agrupado.reset_index()
	return agrupado

# --- Rasgos léxicos y de comportamiento ---
def calcular_rasgos(df_agrupado):
	# Longitud promedio del texto (en palabras)
	df_agrupado['longitud_promedio'] = df_agrupado['text'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
	# Diversidad léxica: palabras únicas / total de palabras
	def diversidad_lexica(texto):
		palabras = texto.split()
		if len(palabras) == 0:
			return 0
		return len(set(palabras)) / len(palabras)
	df_agrupado['diversidad_lexica'] = df_agrupado['text'].apply(diversidad_lexica)
	return df_agrupado

df = None
if opcion == "Subir archivo CSV":
	archivo = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])
	if archivo is not None:
		df = pd.read_csv(archivo)
else:
	df = cargar_ejemplo()

# Mostrar los datos cargados
if df is not None:
	st.subheader("Vista previa de los datos")
	st.dataframe(df)
	# Preprocesamiento
	st.subheader("Preprocesamiento de texto")
	df['text_limpio'] = df['text'].apply(limpiar_texto)
	st.dataframe(df[['user_id', 'text', 'text_limpio']])

	# Agrupamiento por usuario
	st.subheader("Datos agrupados por usuario")
	df_agrupado = agrupar_por_usuario(df)
	st.dataframe(df_agrupado)

	# Cálculo de rasgos léxicos y de comportamiento
	st.subheader("Rasgos léxicos y de comportamiento por usuario")
	df_rasgos = calcular_rasgos(df_agrupado)
	st.dataframe(df_rasgos)

	# Embeddings semánticos con Sentence-BERT/MiniLM
	st.subheader("Embeddings semánticos (Sentence-BERT/MiniLM)")
	with st.spinner("Calculando embeddings semánticos..."):
		modelo = SentenceTransformer('all-MiniLM-L6-v2')
		textos_usuarios = df_rasgos['text'].tolist()
		embeddings = modelo.encode(textos_usuarios)
		# Guardar la dimensión del embedding para mostrar
		st.write(f"Dimensión del embedding: {embeddings.shape[1]}")
		# Mostrar los primeros valores del embedding del primer usuario como ejemplo
		st.write("Ejemplo de embedding para el primer usuario:")
		st.write(embeddings[0][:10])  # Solo los primeros 10 valores para no saturar la vista
	# Puedes guardar los embeddings en el DataFrame si lo necesitas para el siguiente paso
	df_rasgos['embedding'] = list(embeddings)

	# --- Detección de anomalías ---
	st.subheader("Detección de anomalías (Isolation Forest)")
	# Seleccionar rasgos numéricos y embeddings
	rasgos_numericos = df_rasgos[['likes', 'replies', 'num_posts', 'longitud_promedio', 'diversidad_lexica']].values
	embeddings_matrix = np.vstack(df_rasgos['embedding'].values)
	X = np.hstack([rasgos_numericos, embeddings_matrix])

	# Entrenar Isolation Forest
	modelo_iso = IsolationForest(contamination=0.2, random_state=42)
	modelo_iso.fit(X)
	scores = modelo_iso.decision_function(X)
	anomalía = modelo_iso.predict(X)
	# -1 es anómalo, 1 es normal
	df_rasgos['anomaly_score'] = -scores  # Mayor score = más anómalo
	df_rasgos['es_sospechoso'] = (anomalía == -1)

	# Mostrar ranking de sospechosos
	st.subheader("Ranking de usuarios sospechosos")
	ranking = df_rasgos.sort_values('anomaly_score', ascending=False)
	st.dataframe(ranking[['user_id', 'anomaly_score', 'es_sospechoso', 'likes', 'replies', 'num_posts', 'longitud_promedio', 'diversidad_lexica']])

	# --- Visualización interactiva con Plotly ---
	st.subheader("Visualización de rasgos y anomalías")
	# Normalizar anomaly_score para que sea positivo y adecuado para size
	min_score = ranking['anomaly_score'].min()
	size_score = ranking['anomaly_score'] - min_score + 0.1  # Sumar 0.1 para evitar ceros
	fig1 = px.scatter(
		ranking,
		x='longitud_promedio',
		y='diversidad_lexica',
		color='es_sospechoso',
		size=size_score,
		hover_data=['user_id', 'likes', 'replies', 'num_posts'],
		title='Usuarios: Longitud promedio vs Diversidad léxica'
	)
	st.plotly_chart(fig1, use_container_width=True)

	fig2 = px.histogram(
		ranking,
		x='anomaly_score',
		color='es_sospechoso',
		nbins=20,
		title='Distribución de puntajes de anomalía'
	)
	st.plotly_chart(fig2, use_container_width=True)

	# --- Descarga de resultados ---
	st.subheader("Descargar resultados")
	csv = ranking.to_csv(index=False).encode('utf-8')
	st.download_button(
		label="Descargar ranking en CSV",
		data=csv,
		file_name='ranking_sospechosos.csv',
		mime='text/csv'
	)

	# --- Justificación del análisis ---
	st.subheader("Justificación y explicación del análisis")
	st.markdown("""
	- Los usuarios marcados como **sospechosos** presentan comportamientos o contenidos atípicos según los rasgos analizados.
	- Puedes explorar los gráficos para ver cómo se diferencian en longitud de texto, diversidad léxica y otras variables.
	- El puntaje de anomalía es mayor para quienes más se alejan del comportamiento general.
	- Revisa los usuarios con puntaje más alto y analiza si sus publicaciones justifican la sospecha de automatización.
	""")
else:
	st.info("Por favor, sube un archivo CSV o usa el dataset de ejemplo.")
