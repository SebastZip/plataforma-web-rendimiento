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
base_path = os.path.join(os.path.dirname(__file__), "modelos")
modelo = joblib.load(os.path.join(base_path, "modelo_entrenado.pkl"))
scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
columnas_seleccionadas = joblib.load(os.path.join(base_path, "top_12_features.pkl"))

print("Columnas utilizadas por el modelo:")
print(columnas_seleccionadas)

# 3. Mapeo de columnas de Supabase → nombres del modelo
supabase_to_model = {
    "previous_sgpa": "What was your previous SGPA?",
    "credit_completed": "How many Credit did you have completed?",
    "current_semester": "Current Semester",
    "monthly_income": "What is your monthly family income?",
    "average_attendance": "Average attendance on class",
    "social_media_time": "How many hour do you spent daily in social media?",
    "hsc_year": "H.S.C passing year",
    "ever_probation": "Did you ever fall in probation?",
    "skill_time": "How many hour do you spent daily on your skill development?",
    "age": "Age",
    "study_time": "How many hour do you study daily?",
    "scholarship": "Do you have meritorious scholarship ?"
}

# Invertir el diccionario
model_to_supabase = {v: k for k, v in supabase_to_model.items()}

# 4. Inicializar FastAPI
app = FastAPI()

# 5. CORS para desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción: restringe al dominio del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 6. Ruta de predicción por código de estudiante
@app.get("/predecir/{codigo_estudiante}")
def predecir_cgpa(codigo_estudiante: str):
    response = supabase.table("predicciones_estudiantes").select("*").eq("codigo_estudiante", codigo_estudiante).execute()

    if not response.data:
        raise HTTPException(status_code=404, detail="Estudiante no encontrado")

    fila = response.data[0]  # Se espera una fila
    
    print("Columnas esperadas por el modelo:")
    print(columnas_seleccionadas)
    print("Datos recibidos desde Supabase:")
    print(fila)
    
    try:
        # Extraer valores en el orden de las columnas del modelo
        valores_modelo = []
        for col_model in columnas_seleccionadas:
            if col_model not in model_to_supabase:
                raise HTTPException(status_code=400, detail=f"Columna no mapeada: {col_model}")
            col_supabase = model_to_supabase[col_model]
            if col_supabase not in fila:
                raise HTTPException(status_code=400, detail=f"Falta columna en Supabase: {col_supabase}")
            valores_modelo.append(fila[col_supabase])

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error recolectando características: {str(e)}")

    try:
        # Escalar y predecir CGPA
        vector = np.array(valores_modelo).reshape(1, -1)
        vector_esc = scaler.transform(vector)
        pred = modelo.predict(vector_esc)
        pred_float = float(pred.flatten()[0])

        return {
            "codigo_estudiante": codigo_estudiante,
            "cgpa_predicho": round(pred_float, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {str(e)}")
