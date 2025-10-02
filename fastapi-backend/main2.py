# fastapi-backend/main2.py
# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import os

from supabase import create_client, Client

# ---------- Supabase ----------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://ghpihkczhzoaydrceflm.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdocGloa2N6aHpvYXlkcmNlZmxtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDUwNTMzNjIsImV4cCI6MjA2MDYyOTM2Mn0.RUo6qktKAQcmgBYWLa0Lq_pE1UdLB2KS1nWLxr-HaIM")
SUPABASE_TABLE = "predicciones_estudiantes"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------- Artefactos ----------
BASE = Path(__file__).parent / "modelos"

# Regresión
REG_MODEL   = BASE / "modelo_entrenado2.pkl"
REG_SCALER  = BASE / "scaler2.pkl"
REG_IMPUTER = BASE / "reg_imputer.pkl"
REG_FEATS   = BASE / "top_12_features2.pkl"

# Continuidad
CLS_MODEL   = BASE / "continuidad_modelo.pkl"
CLS_SCALER  = BASE / "continuidad_scaler.pkl"
CLS_IMPUTER = BASE / "continuidad_imputer.pkl"
CLS_FEATS   = BASE / "continuidad_features.pkl"
CLS_THR     = BASE / "continuidad_umbral.pkl"

# ---------- Cargar ----------
try:
    reg_model   = joblib.load(REG_MODEL)
    reg_scaler  = joblib.load(REG_SCALER)
    reg_imputer = joblib.load(REG_IMPUTER)
    reg_feats   = joblib.load(REG_FEATS)
except Exception as e:
    raise RuntimeError(f"[Regresión] No pude cargar artefactos: {e}")

try:
    cls_model   = joblib.load(CLS_MODEL)
    cls_scaler  = joblib.load(CLS_SCALER)
    cls_imputer = joblib.load(CLS_IMPUTER)
    cls_feats   = joblib.load(CLS_FEATS)
    cls_thr     = float(joblib.load(CLS_THR))
except Exception as e:
    raise RuntimeError(f"[Clasificación] No pude cargar artefactos: {e}")

# ---------- App ----------
app = FastAPI(title="API Rendimiento Académico", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/")
def root():
    return {"ok": True, "msg": "API viva", "endpoints": ["/health", "/predict/..."]}

# ---------- Mapeo Supabase → modelo ----------
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

def frame_from_payload(payload: Dict[str, Any], required_cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([payload], dtype=object)
    df = normalize_bools(df)
    df = to_numeric(df)
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[required_cols].copy()

def payload_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for supa_key, model_key in SUPABASE_TO_MODEL.items():
        if supa_key in row:
            out[model_key] = row[supa_key]
    return out

def upsert_regresion(codigo: str, sem_act: int, yhat: float):
    supabase.table(SUPABASE_TABLE).update({
        "promedio_predicho": round(float(yhat), 2)
    }).eq("codigo_estudiante", codigo).eq("semestre_actual", sem_act).execute()

def upsert_continuidad(codigo: str, sem_act: int, prob: float, riesgo: int):
    supabase.table(SUPABASE_TABLE).update({
        "prob_riesgo_no_continuar": round(float(prob), 5),
        "riesgo_no_continuar": int(riesgo)
    }).eq("codigo_estudiante", codigo).eq("semestre_actual", sem_act).execute()

# ---------- Schemas ----------
class Input(BaseModel):
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
    estado_observado: Optional[bool | str] = Field(None)
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
        "classification_threshold": cls_thr
    }

# ---------- Predicción JSON ----------
@app.post("/predict/regresion")
def predict_regresion(inp: Input):
    try:
        df = frame_from_payload(inp.model_dump(), reg_feats)
        X  = reg_imputer.transform(df.values)
        X  = reg_scaler.transform(X)
        y  = float(reg_model.predict(X)[0])
        y  = float(np.clip(y, 0.0, 20.0))
        return {"promedio_predicho": round(y, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error regresión: {e}")

@app.post("/predict/continuidad")
def predict_continuidad(inp: Input):
    try:
        df = frame_from_payload(inp.model_dump(), cls_feats)
        X  = cls_imputer.transform(df.values)
        X  = cls_scaler.transform(X)
        p  = float(cls_model.predict_proba(X)[0, 1])
        riesgo = int(p >= cls_thr)
        return {"prob_riesgo": round(p, 5), "umbral": round(cls_thr, 3), "riesgo": riesgo}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error clasificación: {e}")

# ---------- Predicción POR CÓDIGO ----------
@app.get("/predict/regresion/{codigo}")
def predict_regresion_codigo(codigo: str, save: bool = Query(False)):
    res = supabase.table(SUPABASE_TABLE).select("*").eq("codigo_estudiante", codigo)\
          .order("fecha_prediccion", desc=True).limit(1).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Estudiante no encontrado")
    row = res.data[0]
    df  = frame_from_payload(payload_from_row(row), reg_feats)
    X   = reg_imputer.transform(df.values)
    X   = reg_scaler.transform(X)
    y   = float(reg_model.predict(X)[0])
    y   = float(np.clip(y, 0.0, 20.0))
    if save and row.get("semestre_actual") is not None:
        upsert_regresion(codigo, int(row["semestre_actual"]), y)
    return {"codigo_estudiante": codigo, "promedio_predicho": round(y, 2)}

@app.get("/predict/continuidad/{codigo}")
def predict_continuidad_codigo(codigo: str, save: bool = Query(False)):
    res = supabase.table(SUPABASE_TABLE).select("*").eq("codigo_estudiante", codigo)\
          .order("fecha_prediccion", desc=True).limit(1).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Estudiante no encontrado")
    row = res.data[0]
    df  = frame_from_payload(payload_from_row(row), cls_feats)
    X   = cls_imputer.transform(df.values)
    X   = cls_scaler.transform(X)
    p   = float(cls_model.predict_proba(X)[0, 1])
    riesgo = int(p >= cls_thr)
    if save and row.get("semestre_actual") is not None:
        upsert_continuidad(codigo, int(row["semestre_actual"]), p, riesgo)
    return {"codigo_estudiante": codigo, "prob_riesgo": round(p, 5), "umbral": round(cls_thr, 3), "riesgo": riesgo}
