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
import json
import os

from supabase import create_client, Client

# =========================================================
# Supabase (usa variables de entorno en prod)
# =========================================================
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://ghpihkczhzoaydrceflm.supabase.co")
SUPABASE_KEY = os.getenv(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdocGloa2N6aHpvYXlkcmNlZmxtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDUwNTMzNjIsImV4cCI6MjA2MDYyOTM2Mn0.RUo6qktKAQcmgBYWLa0Lq_pE1UdLB2KS1nWLxr-HaIM",
)
SUPABASE_TABLE = "predicciones_estudiantes"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================================================
# Artefactos de modelos (pipelines completos) + metadatos
# =========================================================
BASE = Path(__file__).parent
MODELS_DIR = (BASE.parent / "modelo-predictivo" / "models").resolve()

REG_PIPE_PATH = MODELS_DIR / "best_regressor.joblib"
CLS_PIPE_PATH = MODELS_DIR / "best_classifier.joblib"
REG_META_PATH = MODELS_DIR / "used_features_reg.json"
CLS_META_PATH = MODELS_DIR / "used_features_clf.json"

if not REG_PIPE_PATH.exists() or not REG_META_PATH.exists():
    raise RuntimeError(f"[Regresión] Faltan artefactos: {REG_PIPE_PATH} o {REG_META_PATH}")
if not CLS_PIPE_PATH.exists() or not CLS_META_PATH.exists():
    raise RuntimeError(f"[Clasificación] Faltan artefactos: {CLS_PIPE_PATH} o {CLS_META_PATH}")

try:
    reg_pipe = joblib.load(REG_PIPE_PATH)
except Exception as e:
    raise RuntimeError(f"[Regresión] No pude cargar {REG_PIPE_PATH.name}: {e}")

try:
    cls_pipe = joblib.load(CLS_PIPE_PATH)
except Exception as e:
    raise RuntimeError(f"[Clasificación] No pude cargar {CLS_PIPE_PATH.name}: {e}")

meta_reg = json.loads(REG_META_PATH.read_text(encoding="utf-8"))
meta_clf = json.loads(CLS_META_PATH.read_text(encoding="utf-8"))

REG_COLS = list(meta_reg.get("selected_columns", []))
CLF_COLS = list(meta_clf.get("selected_columns", []))
REG_NUM  = set(meta_reg.get("selected_num", []))
REG_CAT  = set(meta_reg.get("selected_cat", []))
CLF_NUM  = set(meta_clf.get("selected_num", []))
CLF_CAT  = set(meta_clf.get("selected_cat", []))

if not REG_COLS or not CLF_COLS:
    raise RuntimeError("Metadatos sin 'selected_columns'. Reentrena con el script actualizado.")

# Umbral por defecto para clasificación
DEFAULT_CLS_THRESHOLD = float(os.getenv("CLS_THRESHOLD", "0.5"))

# =========================================================
# FastAPI
# =========================================================
app = FastAPI(title="API Rendimiento Académico", version="3.3")

ALLOWED_ORIGINS = [
    "https://plataforma-web-rendimiento.vercel.app",
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400,
)

@app.get("/")
def root():
    return {"ok": True, "msg": "API viva", "endpoints": ["/health", "/meta/features", "/predict/..."]}

# =========================================================
# Helpers de preprocesamiento (inferencias)
# =========================================================
CAT_PLACEHOLDER = "DESCONOCIDO"

BIN_MAP = {"Sí": 1, "Si": 1, "SÍ": 1, "sí": 1, "si": 1, True: 1,
           "No": 0, "NO": 0, "no": 0, False: 0}

PROGRAMAS_VALIDOS = {
    "e.p. de ingenieria de sistemas": "E.P. de Ingenieria de Sistemas",
    "e.p. de ingeniería de sistemas": "E.P. de Ingenieria de Sistemas",
    "e.p. de ingenieria de software": "E.P. de Ingenieria de Software",
    "e.p. de ingeniería de software": "E.P. de Ingenieria de Software",
}

def canonicalize_programa(v: Any) -> str:
    if v is None:
        return CAT_PLACEHOLDER
    s = str(v).strip().lower()
    return PROGRAMAS_VALIDOS.get(s, CAT_PLACEHOLDER)

def canonicalize_sexo(v: Any) -> str:
    if v is None:
        return CAT_PLACEHOLDER
    s = str(v).strip().lower()
    if s in {"m", "masculino"}:
        return "M"
    if s in {"f", "femenino"}:
        return "F"
    return CAT_PLACEHOLDER

def normalize_bools(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "bool":
            df[c] = df[c].map(BIN_MAP).where(df[c].isin(BIN_MAP), df[c])
    return df

def to_numeric_cols(df: pd.DataFrame, numeric_cols: set[str]) -> pd.DataFrame:
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def to_categorical_cols(df: pd.DataFrame, categorical_cols: set[str]) -> pd.DataFrame:
    for c in categorical_cols:
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                    .str.strip()
                    .replace({"": CAT_PLACEHOLDER, "None": CAT_PLACEHOLDER, "nan": CAT_PLACEHOLDER})
            )
    return df

def _ensure_required_columns(df: pd.DataFrame, required_cols: list[str],
                             numeric_cols: set[str], categorical_cols: set[str]) -> pd.DataFrame:
    df = df.copy()
    for col in required_cols:
        if col not in df.columns:
            if col in numeric_cols:
                df[col] = np.nan
            elif col in categorical_cols:
                df[col] = CAT_PLACEHOLDER
            else:
                df[col] = np.nan
    return df[required_cols]

def enrich_and_align(payload: Dict[str, Any],
                     required_cols: list[str],
                     numeric_cols: set[str],
                     categorical_cols: set[str]) -> pd.DataFrame:
    """
    - NO usa 'semestre_actual'. Se trabaja solo con 'anio_ciclo_est'
    - Normaliza programa (2 opciones) y sexo (M/F)
    - Crea edad_en_ingreso si falta (NaN -> se imputará)
    - Tipifica y ordena exactamente como el modelo espera
    """
    data = dict(payload) if payload else {}

    # Categóricas clave: programa y sexo (normalizadas)
    data["programa"] = canonicalize_programa(data.get("programa"))
    data["sexo"]     = canonicalize_sexo(data.get("sexo"))

    # Num opcional: edad_en_ingreso (si falta, se imputará)
    if "edad_en_ingreso" not in data:
        data["edad_en_ingreso"] = None

    df = pd.DataFrame([data], dtype=object)
    df = normalize_bools(df)
    df = to_numeric_cols(df, numeric_cols)
    df = to_categorical_cols(df, categorical_cols)
    df = _ensure_required_columns(df, required_cols, numeric_cols, categorical_cols)
    return df

def validate_required_if_present(model_cols: list[str], df: pd.DataFrame):
    """
    Valida campos que definiste como obligatorios si están en selected_columns del modelo.
    """
    # edad_en_ingreso obligatorio si el modelo la requiere
    if "edad_en_ingreso" in model_cols and pd.isna(df.loc[0, "edad_en_ingreso"]):
        raise HTTPException(status_code=400, detail="Falta el campo obligatorio: edad_en_ingreso")

    # programa obligatorio/restringido (si el modelo lo usa)
    if "programa" in model_cols:
        val = df.loc[0, "programa"]
        if val not in PROGRAMAS_VALIDOS.values() and val != CAT_PLACEHOLDER:
            raise HTTPException(status_code=400, detail="Programa inválido. Use 'E.P. de Ingenieria de Sistemas' o 'E.P. de Ingenieria de Software'.")

    # sexo obligatorio M/F (si el modelo lo usa)
    if "sexo" in model_cols:
        val = df.loc[0, "sexo"]
        if val not in {"M", "F", CAT_PLACEHOLDER}:
            raise HTTPException(status_code=400, detail="Sexo inválido. Solo se admite 'M' o 'F'.")

# =========================================================
# Supabase helpers (solo con anio_ciclo_est)
# =========================================================
def upsert_regresion(codigo: str, ciclo_est: int, yhat: float):
    supabase.table(SUPABASE_TABLE).update({
        "promedio_predicho": round(float(yhat), 2)
    }).eq("codigo_estudiante", codigo).eq("anio_ciclo_est", ciclo_est).execute()

def upsert_continuidad(codigo: str, ciclo_est: int, prob: float, riesgo: int):
    supabase.table(SUPABASE_TABLE).update({
        "prob_riesgo_no_continuar": round(float(prob), 5),
        "riesgo_no_continuar": int(riesgo)
    }).eq("codigo_estudiante", codigo).eq("anio_ciclo_est", ciclo_est).execute()

def get_last_row_by_codigo(codigo: str) -> Dict[str, Any]:
    res = supabase.table(SUPABASE_TABLE).select("*") \
        .eq("codigo_estudiante", codigo) \
        .order("fecha_prediccion", desc=True).limit(1).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Estudiante no encontrado")
    return res.data[0]

# =========================================================
# Schemas (solo lo necesario; el backend completa lo demás)
# =========================================================
class Input(BaseModel):
    # Obligatorios del formulario (nombres exactos)
    promedio_ultima_matricula: Optional[float] = Field(None)
    anio_ciclo_est: Optional[int] = Field(None)              # UI: "Semestre actual"
    programa: Optional[str] = Field(None)                    # Solo 2 opciones
    num_periodo_acad_matric: Optional[int] = Field(None)
    edad_en_ingreso: Optional[float] = Field(None)           # Obligatorio (modelo la usa)
    anio_ingreso: Optional[int] = Field(None)
    sexo: Optional[str] = Field(None)                        # "M" o "F"

    # Otros campos opcionales que puedes guardar (no usados por el modelo actual)
    asistencia_promedio_pct: Optional[float] = Field(None)
    horas_estudio_diarias: Optional[float] = Field(None)
    horas_habilidades_diarias: Optional[float] = Field(None)
    ingreso_familiar_mensual_soles: Optional[float] = Field(None)
    ultimo_periodo_matriculado: Optional[int] = Field(None)  # AAAAT (opcional/BD)

# =========================================================
# Health + Meta
# =========================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "regressor_loaded": REG_PIPE_PATH.name,
        "classifier_loaded": CLS_PIPE_PATH.name,
        "reg_columns": REG_COLS,
        "clf_columns": CLF_COLS,
        "classification_threshold": DEFAULT_CLS_THRESHOLD,
    }

@app.get("/meta/features")
def get_features_meta():
    return {
        "regression": meta_reg,
        "classification": meta_clf,
    }

# =========================================================
# Predicción JSON (payload directo del form)
# =========================================================
@app.post("/predict/regresion")
def predict_regresion(inp: Input):
    try:
        payload = inp.model_dump()
        df = enrich_and_align(payload, REG_COLS, REG_NUM, REG_CAT)
        validate_required_if_present(REG_COLS, df)

        y = float(reg_pipe.predict(df)[0])
        y = float(np.clip(y, 0.0, 20.0))

        faltantes = [c for c in REG_COLS if c not in payload.keys()]
        sobrantes = [c for c in payload.keys() if c not in REG_COLS]
        return {
            "promedio_predicho": round(y, 2),
            "info_columnas": {"faltantes_rellenadas": faltantes, "ignoradas_por_modelo": sobrantes},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error regresión: {e}")

@app.post("/predict/continuidad")
def predict_continuidad(inp: Input, thr: Optional[float] = Query(None)):
    try:
        payload = inp.model_dump()
        df = enrich_and_align(payload, CLF_COLS, CLF_NUM, CLF_CAT)
        validate_required_if_present(CLF_COLS, df)

        if hasattr(cls_pipe, "predict_proba"):
            p = float(cls_pipe.predict_proba(df)[0, 1])
        elif hasattr(cls_pipe, "decision_function"):
            score = float(cls_pipe.decision_function(df)[0])
            p = 1.0 / (1.0 + np.exp(-score))
        else:
            pred = int(cls_pipe.predict(df)[0])
            p = 0.75 if pred == 1 else 0.25

        thr_final = float(thr) if thr is not None else DEFAULT_CLS_THRESHOLD
        riesgo = int(p >= thr_final)

        faltantes = [c for c in CLF_COLS if c not in payload.keys()]
        sobrantes = [c for c in payload.keys() if c not in CLF_COLS]
        return {
            "prob_riesgo": round(p, 5),
            "umbral": round(thr_final, 3),
            "riesgo": riesgo,
            "info_columnas": {"faltantes_rellenadas": faltantes, "ignoradas_por_modelo": sobrantes},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error clasificación: {e}")

# =========================================================
# Predicción POR CÓDIGO (lee supabase, predice y opcionalmente guarda)
# =========================================================
@app.get("/predict/regresion/{codigo}")
def predict_regresion_codigo(codigo: str, save: bool = Query(False)):
    row = get_last_row_by_codigo(codigo)
    df = enrich_and_align(row, REG_COLS, REG_NUM, REG_CAT)
    validate_required_if_present(REG_COLS, df)

    y = float(reg_pipe.predict(df)[0])
    y = float(np.clip(y, 0.0, 20.0))
    if save and row.get("anio_ciclo_est") is not None:
        upsert_regresion(codigo, int(row["anio_ciclo_est"]), y)
    return {"codigo_estudiante": codigo, "promedio_predicho": round(y, 2)}

@app.get("/predict/continuidad/{codigo}")
def predict_continuidad_codigo(codigo: str, save: bool = Query(False), thr: Optional[float] = Query(None)):
    row = get_last_row_by_codigo(codigo)
    df = enrich_and_align(row, CLF_COLS, CLF_NUM, CLF_CAT)
    validate_required_if_present(CLF_COLS, df)

    if hasattr(cls_pipe, "predict_proba"):
        p = float(cls_pipe.predict_proba(df)[0, 1])
    elif hasattr(cls_pipe, "decision_function"):
        score = float(cls_pipe.decision_function(df)[0])
        p = 1.0 / (1.0 + np.exp(-score))
    else:
        pred = int(cls_pipe.predict(df)[0])
        p = 0.75 if pred == 1 else 0.25

    thr_final = float(thr) if thr is not None else DEFAULT_CLS_THRESHOLD
    riesgo = int(p >= thr_final)

    if save and row.get("anio_ciclo_est") is not None:
        upsert_continuidad(codigo, int(row["anio_ciclo_est"]), p, riesgo)

    return {
        "codigo_estudiante": codigo,
        "prob_riesgo": round(p, 5),
        "umbral": round(thr_final, 3),
        "riesgo": riesgo,
    }
