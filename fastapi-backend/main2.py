# main.py
# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import os

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://plataforma-web-rendimiento.vercel.app,http://localhost:5173,http://localhost:3000"
).split(",")

# ---------- Supabase ----------
from supabase import create_client, Client
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://ghpihkczhzoaydrceflm.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdocGloa2N6aHpvYXlkcmNlZmxtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDUwNTMzNjIsImV4cCI6MjA2MDYyOTM2Mn0.RUo6qktKAQcmgBYWLa0Lq_pE1UdLB2KS1nWLxr-HaIM")
SUPABASE_TABLE = "predicciones_estudiantes"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------- Artefactos ----------
BASE_MODELS = Path(__file__).parent / "modelos"

# Regresión (promedio 0–20)
REG_MODEL_PATH  = BASE_MODELS / "modelo_entrenado2.pkl"
REG_SCALER_PATH = BASE_MODELS / "scaler2.pkl"
REG_FEATS_PATH  = BASE_MODELS / "top_12_features2.pkl"

# Clasificación (continuidad: 1 = No continúa)
CLS_MODEL_PATH  = BASE_MODELS / "continuidad_modelo.pkl"
CLS_SCALER_PATH = BASE_MODELS / "continuidad_scaler.pkl"
CLS_FEATS_PATH  = BASE_MODELS / "continuidad_features.pkl"
CLS_THR_PATH    = BASE_MODELS / "continuidad_umbral.pkl"

# ---------- Cargar artefactos ----------
try:
    reg_model  = joblib.load(REG_MODEL_PATH)
    reg_scaler = joblib.load(REG_SCALER_PATH)
    reg_feats  = joblib.load(REG_FEATS_PATH)
except Exception as e:
    raise RuntimeError(f"[Regresión] No pude cargar artefactos: {e}")

try:
    cls_model  = joblib.load(CLS_MODEL_PATH)
    cls_scaler = joblib.load(CLS_SCALER_PATH)
    cls_feats  = joblib.load(CLS_FEATS_PATH)
    cls_thr    = joblib.load(CLS_THR_PATH)
except Exception as e:
    raise RuntimeError(f"[Clasificación] No pude cargar artefactos: {e}")

# ---------- App ----------
app = FastAPI(title="API Rendimiento Académico (Regresión + Continuidad)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=True,                 # si usas cookies/headers de credencial
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],        # opcional: headers que expones al front
    max_age=86400                           # cache del preflight (opcional)
)

# ---------- Mapeo Supabase → columnas de entrenamiento ----------
SUPABASE_TO_MODEL = {
    "sgpa_previo": "sgpa_previo",
    "cgpa_actual": "cgpa_actual",
    "creditos_completados": "creditos_completados",
    "semestre_actual": "semestre_actual",
    "asistencia_promedio_pct": "asistencia_promedio_pct",
    "horas_estudio_diarias": "horas_estudio_diarias",
    "horas_redes_diarias": "horas_redes_diarias",
    "horas_habilidades_diarias": "horas_habilidades_diarias",
    "ingreso_familiar_mensual_soles": "ingreso_familiar_mensual_soles",
    "edad": "edad",
    "estado_observado": "estado_observado",
    "desaprobo_alguna_asignatura": "desaprobo_alguna_asignatura",
    "beca_subvencion_economica": "beca_subvencion_economica",
    "planea_matricularse_prox_ciclo": "planea_matricularse_prox_ciclo",
    "anio_egreso_secundaria": "anio_egreso_secundaria",
}

# ---------- Utils ----------
BIN_MAP = {"Sí": 1, "Si": 1, "SÍ": 1, "sí": 1, "si": 1, True: 1,
           "No": 0, "NO": 0, "no": 0, False: 0}

def normalize_bools(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "bool":
            df[c] = df[c].map(BIN_MAP).where(df[c].isin(BIN_MAP), df[c])
    return df

def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def make_frame(payload: Dict[str, Any], required_cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([payload], dtype=object)
    df = normalize_bools(df)
    df = to_numeric(df)
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[required_cols].copy()
    if df.isnull().any().any():
        df = df.fillna(df.median(numeric_only=True))
    return df

def row_to_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for supa_key, model_key in SUPABASE_TO_MODEL.items():
        if supa_key in row:
            out[model_key] = row[supa_key]
    return out

def upsert_results_regresion(codigo: str, semestre_actual: int, promedio: float):
    supabase.table(SUPABASE_TABLE).update({
        "promedio_predicho": float(round(promedio, 2))
    }).eq("codigo_estudiante", codigo).eq("semestre_actual", semestre_actual).execute()

def upsert_results_continuidad(codigo: str, semestre_actual: int, prob: float, riesgo: int):
    supabase.table(SUPABASE_TABLE).update({
        "prob_riesgo_no_continuar": float(round(prob, 5)),
        "riesgo_no_continuar": int(riesgo)
    }).eq("codigo_estudiante", codigo).eq("semestre_actual", semestre_actual).execute()

# ---------- Schemas ----------
class InputGenerico(BaseModel):
    # incluye todo lo que podrías recibir (el modelo solo usará lo necesario)
    sgpa_previo: Optional[float] = Field(None)
    cgpa_actual: Optional[float] = Field(None)
    creditos_completados: Optional[float] = Field(None)
    semestre_actual: Optional[int] = Field(None)
    asistencia_promedio_pct: Optional[float] = Field(None)
    horas_estudio_diarias: Optional[float] = Field(None)
    horas_redes_diarias: Optional[float] = Field(None)
    horas_habilidades_diarias: Optional[float] = Field(None)
    ingreso_familiar_mensual_soles: Optional[float] = Field(None)
    edad: Optional[int] = Field(None)
    estado_observado: Optional[bool | str] = Field(None, description="Sí/No o true/false")
    desaprobo_alguna_asignatura: Optional[bool | str] = Field(None)
    beca_subvencion_economica: Optional[bool | str] = Field(None)
    planea_matricularse_prox_ciclo: Optional[bool | str] = Field(None)
    anio_egreso_secundaria: Optional[int] = Field(None)

# ---------- Health ----------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "regression_features": reg_feats,
        "classification_features": cls_feats,
        "classification_threshold": float(cls_thr),
    }

# ---------- PREDICCIÓN (JSON) ----------
@app.post("/predict/regresion")
def predict_regresion(inp: InputGenerico, save: bool = Query(False)):
    try:
        df = make_frame(inp.model_dump(), reg_feats)
        X = reg_scaler.transform(df.values)
        yhat = float(reg_model.predict(X)[0])
        yhat = float(np.clip(yhat, 0.0, 20.0))
        # guardar (opcional)
        if save and inp.semestre_actual is not None:
            # necesitas también el código del estudiante si quieres guardar
            # este endpoint JSON no lo trae; para guardar usa el endpoint por código
            pass
        return {"promedio_predicho": round(yhat, 2), "rango": "0–20", "features_usadas": reg_feats}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en regresión: {e}")

@app.post("/predict/continuidad")
def predict_continuidad(inp: InputGenerico, save: bool = Query(False)):
    try:
        df = make_frame(inp.model_dump(), cls_feats)
        X = cls_scaler.transform(df.values)
        p = float(cls_model.predict_proba(X)[0, 1])  # prob(1 = no continúa)
        riesgo = int(p >= cls_thr)
        # guardar (opcional)
        if save and inp.semestre_actual is not None:
            pass
        return {
            "prob_riesgo": round(p, 5),
            "umbral": round(float(cls_thr), 3),
            "riesgo": riesgo,  # 1 = riesgo de no continuar
            "features_usadas": cls_feats
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en clasificación: {e}")

# ---------- PREDICCIÓN POR CÓDIGO (lee Supabase y puede guardar) ----------
@app.get("/predict/regresion/{codigo}")
def predict_regresion_codigo(codigo: str, save: bool = Query(False)):
    res = supabase.table(SUPABASE_TABLE).select("*").eq("codigo_estudiante", codigo).order("fecha_prediccion", desc=True).limit(1).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Estudiante no encontrado")
    row = res.data[0]
    payload = row_to_payload(row)
    df = make_frame(payload, reg_feats)
    try:
        X = reg_scaler.transform(df.values)
        yhat = float(reg_model.predict(X)[0])
        yhat = float(np.clip(yhat, 0.0, 20.0))
        if save and "semestre_actual" in row and row["semestre_actual"] is not None:
            upsert_results_regresion(codigo, int(row["semestre_actual"]), yhat)
        return {"codigo_estudiante": codigo, "promedio_predicho": round(yhat, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en regresión por código: {e}")

@app.get("/predict/continuidad/{codigo}")
def predict_continuidad_codigo(codigo: str, save: bool = Query(False)):
    res = supabase.table(SUPABASE_TABLE).select("*").eq("codigo_estudiante", codigo).order("fecha_prediccion", desc=True).limit(1).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Estudiante no encontrado")
    row = res.data[0]
    payload = row_to_payload(row)
    df = make_frame(payload, cls_feats)
    try:
        X = cls_scaler.transform(df.values)
        p = float(cls_model.predict_proba(X)[0, 1])
        riesgo = int(p >= cls_thr)
        if save and "semestre_actual" in row and row["semestre_actual"] is not None:
            upsert_results_continuidad(codigo, int(row["semestre_actual"]), p, riesgo)
        return {
            "codigo_estudiante": codigo,
            "prob_riesgo": round(p, 5),
            "umbral": round(float(cls_thr), 3),
            "riesgo": riesgo
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en continuidad por código: {e}")
