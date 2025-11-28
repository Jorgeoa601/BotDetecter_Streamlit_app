# 📰 News Sentiment Analyzer (ES)

Aplicación Streamlit para la detección de anomalías y bots en la plataforma de youtube. 
Utiliza una combinación de Ingeniería de Rasgos Conductuales y Léxicos junto con el 
modelo de lenguaje BETO para obtener embeddings de texto, que luego son procesados por 
un modelo de Isolation Forest.

## 🚀 Cómo ejecutar

```bash
cd "Taller 3"
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 📥 Modos de entrada
La aplicación está diseñada para procesar datos de comentarios de usuarios que incluyan métricas de actividad:

Dataset de Ejemplo: Utiliza un dataset sintético predefinido para una demostración instantánea.

Subir CSV: Carga tu propio archivo CSV. Debe contener obligatoriamente las siguientes columnas:
user_id (Identificador único del usuario)
text (Contenido del comentario)
timestamp (Marca de tiempo para calcular métricas temporales)
likes (Número de likes/interacciones)
replies (Número de respuestas)

## Como Utilizar Scrapper de Youtube
El scrapper viene configurado para funcionar con solo correr el archivo
lo unico necesario es primero crear un proyecto en Google Cloud Console
habilitar la opcion YouTube Data API v3 y luego generar una API Key, luego almacenarlar en la variable "API_KEY"
por ultimo en la variable "VIDEO_ID" almacenar el ID del video al scrapear.
Ejemplo. link normal = https://www.youtube.com/watch?v=xvFZjo5PgG0&list=RDxvFZjo5PgG0&start_radio=1
ID = xvFZjo5PgG0

## ⚠️ Notas
- La primera ejecución descargará pesos del modelo (requiere internet).
- Umbral: El Risk Score es relativo. Ajuste el slider en la barra lateral para cambiar la sensibilidad de detección y 
ver cómo afecta la clasificación final.
- Descarga de Resultados: Los resultados de la clasificación final (ID de Usuario y Nivel de Riesgo) pueden descargarse 
como CSV al final de la aplicacion.