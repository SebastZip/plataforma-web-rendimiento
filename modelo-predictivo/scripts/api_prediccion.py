from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from supabase import create_client, Client
import os

SUPABASE_URL = os.getenv("SUPABASE_URL") or "https://ghpihkczhzoaydrceflm.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdocGloa2N6aHpvYXlkcmNlZmxtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDUwNTMzNjIsImV4cCI6MjA2MDYyOTM2Mn0.RUo6qktKAQcmgBYWLa0Lq_pE1UdLB2KS1nWLxr-HaIM"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


modelo = joblib.load("../modelos/xgboost_model.joblib")
scaler = joblib.load("../modelos/xgboost_scaler.joblib")


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predecir/{codigo_estudiante}")
def predecir_desde_supabase(codigo_estudiante: str):

    response = supabase.table("predicciones_estudiantes").select("*").eq("codigo_estudiante", codigo_estudiante).execute()

    if not response.data:
        raise HTTPException(status_code=404, detail="Estudiante no encontrado")

    fila = response.data[0]  

 
    columnas_a_excluir = ["codigo_estudiante", "id", "performance"]
    features = [valor for clave, valor in fila.items() if clave not in columnas_a_excluir]

    try:
     
        vector = np.array(features).reshape(1, -1)
        vector_esc = scaler.transform(vector)
        pred = modelo.predict(vector_esc)[0]

        mapa = {0: "Empeoró", 1: "Igual", 2: "Mejoró"}
        return {"codigo_estudiante": codigo_estudiante, "prediccion": int(pred), "resultado": mapa[pred]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")
