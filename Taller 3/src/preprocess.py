# src/preprocess.py
import pandas as pd
import re
import emoji
import unicodedata

def limpiar_texto(texto):
    if pd.isnull(texto): return ""
    texto = str(texto)
    # Normalización Unicode
    texto = unicodedata.normalize("NFKC", texto)
    # Eliminar URLs y menciones
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"@\w+", "", texto)
    # Emojis a texto o eliminar (aquí los eliminamos para BERT puro)
    texto = emoji.replace_emoji(texto, replace="")
    # Limpiar espacios extra
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def calcular_ttr(text_list):
    """Calcula Type-Token Ratio (Riqueza léxica) de todos los textos unidos."""
    full_text = " ".join(text_list)
    tokens = full_text.split()
    if not tokens: return 0
    return len(set(tokens)) / len(tokens)

def detectar_repeticion(text_list):
    """Detecta si el usuario repite exactamente los mismos comentarios."""
    if len(text_list) < 2: return 0
    duplicados = len(text_list) - len(set(text_list))
    return duplicados / len(text_list)