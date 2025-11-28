# src/features.py
import pandas as pd
import numpy as np
import re
from .preprocess import limpiar_texto, calcular_ttr, detectar_repeticion

def procesar_datos(df_input):
    """
    Realiza la limpieza de textos, calcula las métricas conductuales y léxicas,
    y devuelve dos DataFrames: uno agrupado por usuario (df_users) y uno 
    detallado por post (df_processed_posts).
    """
    df = df_input.copy() 
    
    # Limpieza inicial
    df["text_limpio"] = df["text"].apply(limpiar_texto)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # --- A. Rasgos de Comportamiento (Agrupación) ---
    metrics = []
    processed_posts = []

    for user_id, group in df.groupby('user_id'):
        # 1. Temporales
        tiempos = group['timestamp'].sort_values()
        if len(tiempos) > 1:
            diffs = tiempos.diff().dt.total_seconds().dropna() / 60 # Minutos
            intervalo_medio = diffs.mean()
            # Frecuencia: Posts por día (aprox)
            rango_dias = (tiempos.max() - tiempos.min()).days
            frecuencia = len(tiempos) / (rango_dias if rango_dias > 0 else 1)
        else:
            intervalo_medio = 0
            frecuencia = 1 # 1 post en 1 dia hipotético

        # 2. Actividad Nocturna (00:00 - 06:00)
        horas = tiempos.dt.hour
        nocturnidad = ((horas >= 0) & (horas < 6)).sum() / len(tiempos)

        # 3. Interacciones
        avg_likes = group['likes'].mean()
        avg_replies = group['replies'].mean()

        # 4. Léxicos
        textos = group['text_limpio'].tolist()
        ttr = calcular_ttr(textos)
        repeticion = detectar_repeticion(textos)
        avg_len = np.mean([len(t.split()) for t in textos]) if textos else 0

        metrics.append({
            'user_id': user_id,
            'num_posts': len(group),
            'intervalo_medio': intervalo_medio,
            'frecuencia_diaria': frecuencia,
            'nocturnidad': nocturnidad,
            'avg_likes': avg_likes,
            'avg_replies': avg_replies,
            'ttr': ttr, 
            'tasa_repeticion': repeticion, 
            'longitud_promedio': avg_len,
            'textos_raw': textos # Guardamos para embeddings
        })
        
        # Almacenar posts limpios para la tabla didáctica (un post por fila)
        for original_text, clean_text in zip(group['text'], group['text_limpio']):
            processed_posts.append({
                'user_id': user_id,
                'texto_original': original_text,
                'texto_limpio': clean_text,
            })
    
    df_metrics = pd.DataFrame(metrics).fillna(0) # Aplicar fillna(0) aquí
    df_processed_posts = pd.DataFrame(processed_posts)

    return df_metrics, df_processed_posts