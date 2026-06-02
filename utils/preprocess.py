# utils/preprocess.py
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Descargar recursos de NLTK
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# Configurar stopwords y lematizador
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Limpiar y preprocesar texto"""
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens 
              if token not in stop_words and len(token) > 2]
    return ' '.join(tokens)

def extract_rating(rating_text):
    """Extraer el número del rating"""
    if pd.isna(rating_text):
        return None
    match = re.search(r'Rated (\d+) out of 5 stars', str(rating_text))
    if match:
        return int(match.group(1))
    return None

def rating_to_sentiment(rating):
    """Convertir rating a sentimiento"""
    if rating <= 2:
        return 'Negativo'
    elif rating == 3:
        return 'Neutro'
    else:
        return 'Positivo'

def analyze_problems(text, problem_keywords):
    """Identificar qué problemas menciona un texto"""
    if not isinstance(text, str):
        return []
    text_lower = text.lower()
    problems_found = []
    for category, keywords in problem_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            problems_found.append(category)
    return problems_found

# Definición de problemas principales para Casa Tuning
PROBLEM_KEYWORDS = {
    'Atencion al Cliente': [
        'atencion', 'servicio', 'trato', 'personal', 'soporte',
        'respuesta', 'ayuda', 'asesoria', 'comunicacion', 'encargado'
    ],

    'Entrega/Retiro': [
        'entrega', 'retiro', 'recoger', 'recibida', 'demora entrega',
        'moto lista', 'espera entrega', 'retraso entrega'
    ],

    'Tiempo de Servicio': [
        'tarde', 'retraso', 'espera', 'demora', 'lento',
        'tiempo', 'dias', 'semanas', 'no estuvo lista'
    ],

    'Garantia/Reclamos': [
        'garantia', 'reclamo', 'devolucion', 'reembolso',
        'cobro extra', 'devolvieron dinero', 'solucion'
    ],

    'Calidad de Reparacion': [
        'reparacion', 'falla', 'motor', 'mal reparado',
        'defectuoso', 'quedo mal', 'problema continuo',
        'diagnostico incorrecto', 'mal trabajo'
    ],

    'Calidad de Repuestos': [
        'repuesto', 'pieza', 'defectuosa', 'mala calidad',
        'usado', 'generico', 'original', 'danado', 'fallo'
    ],

    'Servicio Electrico': [
        'electrico', 'bateria', 'luces', 'cableado',
        'encendido', 'corto circuito', 'conexion'
    ],

    'Tuning/Personalizacion': [
        'tuning', 'personalizacion', 'modificacion',
        'pintura', 'vinil', 'accesorios', 'instalacion',
        'acabado', 'estetica'
    ]
}