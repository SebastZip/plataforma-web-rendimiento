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

# =========================================================
# Supabase (usa variables de entorno en prod)
# =========================================================
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://ghpihkczhzoaydrceflm.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdocGloa2N6aHpvYXlkcmNlZmxtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDUwNTMzNjIsImV4cCI6MjA2MDYyOTM2Mn0.RUo6qktKAQcmgBYWLa0Lq_pE1UdLB2KS1nWLxr-HaIM")
SUPABASE_TABLE = "predicciones_estudiantes"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================================================
# Artefactos de modelos (pipelines completos)
# =========================================================
BASE = Path(__file__).parent
MODELS_DIR = BASE / "models"

REG_PIPE_PATH = MODELS_DIR / "best_regressor.joblib"
CLS_PIPE_PATH = MODELS_DIR / "best_classifier.joblib"

try:
    reg_pipe = joblib.load(REG_PIPE_PATH)  # Pipeline sklearn completo
except Exception as e:
    raise RuntimeError(f"[Regresión] No pude cargar {REG_PIPE_PATH.name}: {e}")

try:
    cls_pipe = joblib.load(CLS_PIPE_PATH)  # Pipeline sklearn completo
except Exception as e:
    raise RuntimeError(f"[Clasificación] No pude cargar {CLS_PIPE_PATH.name}: {e}")

# Si tu clasificador no trae umbral, define uno por defecto (ej. 0.5)
DEFAULT_CLS_THRESHOLD = float(os.getenv("CLS_THRESHOLD", "0.5"))

# =========================================================
# FastAPI
# =========================================================
app = FastAPI(title="API Rendimiento Académico", version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/")
def root():
    return {"ok": True, "msg": "API viva", "endpoints": ["/health", "/predict/..."]}

# =========================================================
# Mapeo Supabase → features del modelo (nombres del form)
# =========================================================
# Requeridos + opcionales que acordamos (se pueden mandar NA, el pipeline imputa)
SUPABASE_TO_MODEL: Dict[str, str] = {
    # Requeridos (modelo regresión)
    "promedio_ultima_matricula": "promedio_ultima_matricula",
    "semestre_actual": "semestre_actual",
    "num_periodo_acad_matric": "num_periodo_acad_matric",
    "ultimo_periodo_matriculado": "ultimo_periodo_matriculado",
    "anio_ingreso": "anio_ingreso",
    # Otros que quizá use tu pipeline (si los tenía al entrenar)
    "edad": "edad",

    # Opcionales (investigación / futuro retrain)
    "asistencia_promedio_pct": "asistencia_promedio_pct",
    "horas_estudio_diarias": "horas_estudio_diarias",
    "horas_redes_diarias": "horas_redes_diarias",
    "horas_habilidades_diarias": "horas_habilidades_diarias",
    "ingreso_familiar_mensual_soles": "ingreso_familiar_mensual_soles",

    # Binarios
    "estado_observado": "estado_observado",
    "desaprobo_alguna_asignatura": "desaprobo_alguna_asignatura",
    "beca_subvencion_economica": "beca_subvencion_economica",
    "planea_matricularse_prox_ciclo": "planea_matricularse_prox_ciclo",
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

def frame_from_payload(payload: Dict[str, Any], feature_names: list[str] | None = None) -> pd.DataFrame:
    """Construye un DF con columnas del mapeo. Si feature_names se pasa, reordena a ese orden."""
    # map supabase keys -> model keys
    mapped = {model_key: payload.get(supa_key, None) for supa_key, model_key in SUPABASE_TO_MODEL.items()}
    df = pd.DataFrame([mapped], dtype=object)
    df = normalize_bools(df)
    df = to_numeric(df)
    # Asegurar columnas esperadas si nos dan la lista
    if feature_names:
        for col in feature_names:
            if col not in df.columns:
                df[col] = np.nan
        df = df[feature_names]
    return df

def payload_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extrae del row de Supabase solo las columnas que mapeamos a features."""
    out = {}
    for supa_key, model_key in SUPABASE_TO_MODEL.items():
        if supa_key in row:
            out[supa_key] = row[supa_key]
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

# =========================================================
# Schemas (solo los campos que realmente usaremos)
# =========================================================
class Input(BaseModel):
    promedio_ultima_matricula: Optional[float] = Field(None)
    semestre_actual: Optional[int] = Field(None)
    num_periodo_acad_matric: Optional[int] = Field(None)
    ultimo_periodo_matriculado: Optional[int] = Field(None)  # AAAAT
    anio_ingreso: Optional[int] = Field(None)
    edad: Optional[int] = Field(None)

    asistencia_promedio_pct: Optional[float] = Field(None)
    horas_estudio_diarias: Optional[float] = Field(None)
    horas_redes_diarias: Optional[float] = Field(None)
    horas_habilidades_diarias: Optional[float] = Field(None)
    ingreso_familiar_mensual_soles: Optional[float] = Field(None)

    estado_observado: Optional[bool | str] = Field(None)
    desaprobo_alguna_asignatura: Optional[bool | str] = Field(None)
    beca_subvencion_economica: Optional[bool | str] = Field(None)
    planea_matricularse_prox_ciclo: Optional[bool | str] = Field(None)

# =========================================================
# Health
# =========================================================
def get_pipe_feature_names(pipe) -> list[str] | None:
    try:
        # Si el preprocessor es ColumnTransformer / OneHot con get_feature_names_out
        pre = pipe.named_steps.get("pre") or pipe.named_steps.get("columntransformer")
        if pre is not None and hasattr(pre, "get_feature_names_out"):
            # En muchos casos devuelve nombres transformados; igual nos sirve de referencia
            return list(pre.get_feature_names_out())
    except Exception:
        pass
    return None

@app.get("/health")
def health():
    return {
        "status": "ok",
        "regressor_loaded": REG_PIPE_PATH.name,
        "classifier_loaded": CLS_PIPE_PATH.name,
        "reg_pre_features": get_pipe_feature_names(reg_pipe),
        "cls_pre_features": get_pipe_feature_names(cls_pipe),
        "classification_threshold": DEFAULT_CLS_THRESHOLD
    }

# =========================================================
# Predicción JSON (payload)
# =========================================================
@app.post("/predict/regresion")
def predict_regresion(inp: Input):
    try:
        df = frame_from_payload(inp.model_dump())
        y  = float(reg_pipe.predict(df)[0])
        y  = float(np.clip(y, 0.0, 20.0))
        return {"promedio_predicho": round(y, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error regresión: {e}")

@app.post("/predict/continuidad")
def predict_continuidad(inp: Input, thr: Optional[float] = Query(None)):
    try:
        df = frame_from_payload(inp.model_dump())
        # Probabilidad de clase positiva (riesgo)
        if hasattr(cls_pipe, "predict_proba"):
            p = float(cls_pipe.predict_proba(df)[0, 1])
        else:
            # Fallback para modelos sin predict_proba (ej. SVM sin probas)
            if hasattr(cls_pipe, "decision_function"):
                from sklearn.metrics import roc_auc_score
                score = float(cls_pipe.decision_function(df)[0])
                # Normalización simple a [0,1] (no calibrada)
                p = 1.0 / (1.0 + np.exp(-score))
            else:
                pred = int(cls_pipe.predict(df)[0])
                p = 0.75 if pred == 1 else 0.25
        threshold = float(thr) if thr is not None else DEFAULT_CLS_THRESHOLD
        riesgo = int(p >= threshold)
        return {"prob_riesgo": round(p, 5), "umbral": round(threshold, 3), "riesgo": riesgo}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error clasificación: {e}")

# =========================================================
# Predicción POR CÓDIGO (lee supabase, predice, y opcionalmente guarda)
# =========================================================
def get_last_row_by_codigo(codigo: str) -> Dict[str, Any]:
    res = supabase.table(SUPABASE_TABLE).select("*") \
          .eq("codigo_estudiante", codigo) \
          .order("fecha_prediccion", desc=True).limit(1).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Estudiante no encontrado")
    return res.data[0]

@app.get("/predict/regresion/{codigo}")
def predict_regresion_codigo(codigo: str, save: bool = Query(False)):
    row = get_last_row_by_codigo(codigo)
    payload = payload_from_row(row)
    df = frame_from_payload(payload)
    y = float(reg_pipe.predict(df)[0])
    y = float(np.clip(y, 0.0, 20.0))
    if save and row.get("semestre_actual") is not None:
        upsert_regresion(codigo, int(row["semestre_actual"]), y)
    return {"codigo_estudiante": codigo, "promedio_predicho": round(y, 2)}

@app.get("/predict/continuidad/{codigo}")
def predict_continuidad_codigo(codigo: str, save: bool = Query(False), thr: Optional[float] = Query(None)):
    row = get_last_row_by_codigo(codigo)
    payload = payload_from_row(row)
    df = frame_from_payload(payload)

    if hasattr(cls_pipe, "predict_proba"):
        p = float(cls_pipe.predict_proba(df)[0, 1])
    else:
        if hasattr(cls_pipe, "decision_function"):
            score = float(cls_pipe.decision_function(df)[0])
            p = 1.0 / (1.0 + np.exp(-score))
        else:
            pred = int(cls_pipe.predict(df)[0])
            p = 0.75 if pred == 1 else 0.25

    threshold = float(thr) if thr is not None else DEFAULT_CLS_THRESHOLD
    riesgo = int(p >= threshold)

    if save and row.get("semestre_actual") is not None:
        upsert_continuidad(codigo, int(row["semestre_actual"]), p, riesgo)

    return {
        "codigo_estudiante": codigo,
        "prob_riesgo": round(p, 5),
        "umbral": round(threshold, 3),
        "riesgo": riesgo
    }
