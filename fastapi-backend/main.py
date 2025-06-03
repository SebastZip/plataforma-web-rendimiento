#Paso 1: Instalar las dependencias necesarias
#pip install fastapi uvicorn supabase

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from supabase import create_client, Client
import os

# 1. Conexión Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL") or "https://ghpihkczhzoaydrceflm.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdocGloa2N6aHpvYXlkcmNlZmxtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDUwNTMzNjIsImV4cCI6MjA2MDYyOTM2Mn0.RUo6qktKAQcmgBYWLa0Lq_pE1UdLB2KS1nWLxr-HaIM"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 2. Cargar modelo y preprocesamiento
modelo = joblib.load("modelos/catboost_modelo.pkl")
scaler = joblib.load("modelos/escalador.pkl")
columnas_modelo = joblib.load("modelos/columnas_seleccionadas.pkl")

# 3. Diccionario de mapeo: nombres en Supabase → nombres originales del modelo
supabase_to_model = {
    "teacher_consultancy": "Do you attend in teacher consultancy for any kind of academical problems?",
    "study_with_peers": "Do you study with your classmates?",
    "study_duration": "How long do you study on average per day?",
    "ask_questions": "Do you ask the teacher questions when you don't understand?",
    # Añade todos los necesarios aquí
}

# Invertir el diccionario para usarlo en orden
model_to_supabase = {v: k for k, v in supabase_to_model.items()}

# 4. Inicializar FastAPI
app = FastAPI()

# 5. CORS para desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica tu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 6. Ruta de predicción
# Diccionario de mapeo: columnas del modelo -> columnas en Supabase
model_to_supabase = {
    'Current Semester': 'Current Semester',
    'How many hour do you study daily?': 'How many hour do you study daily?',
    'Do you have personal Computer?': 'Do you have personal Computer?',
    'How many hour do you spent daily in social media?': 'How many hour do you spent daily in social media?',
    'Average attendance on class': 'Average attendance on class',
    'Did you ever fall in probation?': 'Did you ever fall in probation?',
    'Do you attend in teacher consultancy for any kind of academical problems?': 'Do you attend in teacher consultancy for any kind of academical',
    'Are you engaged with any co-curriculum activities?': 'Are you engaged with any co-curriculum activities?',
    'With whom you are living with?': 'With whom you are living with?',
    'What was your previous SGPA?': 'What was your previous SGPA?',
    'How many Credit did you have completed?': 'How many Credit did you have completed?',
    'What is your monthly family income?': 'What is your monthly family income?',
    'What is your relationship status?_Married': 'What is your relationship status?_Married'
}

# Cargar nombres de columnas del modelo
columnas_modelo = list(model_to_supabase.keys())

@app.get("/predecir/{codigo_estudiante}")
def predecir_desde_supabase(codigo_estudiante: str):
    response = supabase.table("predicciones_estudiantes").select("*").eq("codigo_estudiante", codigo_estudiante).execute()

    if not response.data:
        raise HTTPException(status_code=404, detail="Estudiante no encontrado")

    fila = response.data[0]  # Se espera una sola fila por código

    try:
        # Extraer los valores usando el diccionario sin importar el orden
        valores_modelo = []
        for columna_modelo in columnas_modelo:
            columna_supabase = model_to_supabase[columna_modelo]
            if columna_supabase not in fila:
                raise HTTPException(status_code=400, detail=f"Falta la columna en Supabase: '{columna_supabase}'")
            valores_modelo.append(fila[columna_supabase])

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al recolectar características: {str(e)}")

    try:
        # Predecir
        vector = np.array(valores_modelo).reshape(1, -1)
        vector_esc = scaler.transform(vector)
        pred = modelo.predict(vector_esc)
        pred = int(pred.flatten()[0])

        mapa = {0: "Empeoró", 1: "Igual", 2: "Mejoró"}
        return {
            "codigo_estudiante": codigo_estudiante,
            "prediccion": int(pred),
            "resultado": mapa.get(pred, "Desconocido")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {str(e)}")