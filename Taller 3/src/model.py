# src/model.py
import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# ================= 1. CARGA EFICIENTE DEL MODELO (CACHÉ) =================
@st.cache_resource
def cargar_modelo_beto():
    """Carga el modelo una sola vez para evitar recargas lentas."""
    return SentenceTransformer("dccuchile/bert-base-spanish-wwm-cased")

def entrenar_y_predecir(df_users, umbral_risk_score):
    """Genera embeddings, entrena Isolation Forest, predice y calcula el Risk Score."""
    
    model = cargar_modelo_beto()
    
    # --- 1. Generación de Embeddings ---
    user_embeddings = []
    with st.spinner('🧠 Generando Embeddings con BETO (esto puede tardar)...'):
        for textos in df_users['textos_raw']:
            if len(textos) > 0:
                vectors = model.encode(textos) 
                user_avg_vector = np.mean(vectors, axis=0)
                user_embeddings.append(user_avg_vector)
            else:
                user_embeddings.append(np.zeros(768))
        
        X_embeddings = np.array(user_embeddings)

    # --- 2. Modelado ---
    # 1. Reducción de dimensiones de Texto (PCA)
    pca = PCA(n_components=2)
    X_text_pca = pca.fit_transform(X_embeddings)
    
    # 2. Selección y Escalado de Features de Conducta
    features_conducta = ['num_posts', 'intervalo_medio', 'frecuencia_diaria', 'nocturnidad', 'ttr', 'tasa_repeticion', 'avg_likes', 'avg_replies', 'longitud_promedio']
    X_conducta = df_users[features_conducta].values
    scaler = StandardScaler()
    X_conducta_scaled = scaler.fit_transform(X_conducta)

    # 3. Isolation Forest (Ensemble: Conducta + Texto)
    X_final = np.hstack([X_conducta_scaled, X_text_pca])
    
    iso_forest = IsolationForest(n_estimators=100, contamination=0.2, random_state=42)
    iso_forest.fit(X_final)
    df_users['anomaly_score'] = iso_forest.predict(X_final)
    df_users['decision_function'] = iso_forest.decision_function(X_final)
    
    # Convertir a Risk Score (0 a 1)
    scaler_score = MinMaxScaler()
    df_users['risk_score'] = 1 - scaler_score.fit_transform(df_users[['decision_function']]) # Invertimos: alto score = alto riesgo
    
    # Aplicar el umbral DINÁMICO del slider
    df_users['es_bot'] = df_users['risk_score'] > umbral_risk_score 

    return df_users, features_conducta