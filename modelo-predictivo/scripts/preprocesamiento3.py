# preprocess.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# ---------- Helper interno (compartido) ----------
def _load_and_clean(ruta_excel):
    ruta = Path(ruta_excel)
    if not ruta.exists():
        raise FileNotFoundError(f"No encuentro el archivo: {ruta.resolve()}")

    df = pd.read_excel(ruta)

    # 1) Quitar columnas no predictoras / textuales
    drop_cols = ["codigo_anonimo", "motivo_no_matricula", "escuela_profesional"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # 2) Mapear binarias Sí/No -> 1/0
    bin_map = {"Sí": 1, "No": 0}
    for col in ["estado_observado", "desaprobo_alguna_asignatura",
                "beca_subvencion_economica", "planea_matricularse_prox_ciclo"]:
        if col in df.columns:
            df[col] = df[col].map(bin_map)

    # 3) Asegurar numéricos
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 4) Drop NA y reindex
    df = df.dropna().reset_index(drop=True)
    return df


# ---------- Regresión: predicción del promedio (HE03) ----------
def preparar_regresion(
    ruta_excel="C:\\Users\\Sebas 2\\Desktop\\plataforma-web-rendimiento\\modelo-predictivo\\dataset\\datos_estudiantes_FISI_peru_sinteticos_1800.xlsx",
    target="promedio_final_ciclo_actual",
    test_size=0.2,
    random_state=42,
    path_out="dataset_regresion_limpio.parquet",
):
    df = _load_and_clean(ruta_excel)

    # Separar X / y
    if target not in df.columns:
        raise KeyError(f"Target '{target}' no está en el dataset.")
    y = df[target].astype(float)
    X = df.drop(columns=[target])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Guardar dataset limpio (auditoría)
    Path(path_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path_out, index=False)

    feature_names = X.columns.tolist()
    return X_train, X_test, y_train, y_test, feature_names


# ---------- Clasificación: deserción / continuidad (HE02) ----------
def preparar_clasificacion(
    ruta_excel="C:\\Users\\Sebas 2\\Desktop\\plataforma-web-rendimiento\\modelo-predictivo\\dataset\\datos_estudiantes_FISI_peru_sinteticos_1800.xlsx",
    target_cls="planea_matricularse_prox_ciclo",  # 1 = Sí, 0 = No
    clase_positiva_es="No",  # para re-mapear si quieres que 1 signifique 'No continúa'
    test_size=0.2,
    random_state=42,
    path_out="dataset_clasificacion_limpio.parquet",
):
    """
    Por defecto usa 'planea_matricularse_prox_ciclo' (1/0). Si quieres que la clase positiva sea 'No continúa',
    establecemos y=1 para 'No' (riesgo de deserción).
    """
    df = _load_and_clean(ruta_excel)

    if target_cls not in df.columns:
        raise KeyError(f"Target '{target_cls}' no está en el dataset.")

    y_raw = df[target_cls].astype(int)  # 1=Sí, 0=No (tras el mapeo)
    # Si la clase positiva debe ser 'No continúa', invertimos:
    if clase_positiva_es.lower().startswith("no"):
        y = (y_raw == 0).astype(int)  # 1 = No continúa
    else:
        y = y_raw.copy()  # 1 = Sí continúa

    X = df.drop(columns=[target_cls])

    # Split ESTRATIFICADO (importante en clases desbalanceadas)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    Path(path_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path_out, index=False)

    feature_names = X.columns.tolist()
    return X_train, X_test, y_train, y_test, feature_names
