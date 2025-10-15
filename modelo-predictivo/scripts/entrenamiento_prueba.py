# scripts/entrenar_modelos_excel.py
# -*- coding: utf-8 -*-
from pathlib import Path
import os
import numpy as np
import pandas as pd
from math import sqrt
import json
import warnings

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.metrics import (
    make_scorer, mean_absolute_error, r2_score, roc_auc_score,
    f1_score, recall_score, get_scorer
)
from sklearn.inspection import permutation_importance

# Regressors (sklearn)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor

# Classifiers (sklearn)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier

import joblib

# ===== Optional libs (LightGBM / CatBoost) =====
HAS_LGBM = True
HAS_CAT  = True
try:
    from lightgbm import LGBMRegressor, LGBMClassifier
except Exception:
    HAS_LGBM = False
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
except Exception:
    HAS_CAT = False

# ===================== Config =====================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "SUM_FISI_10_21_clean.xlsx"
OUT_DIR_DS = BASE_DIR / "dataset"
OUT_DIR_MD = BASE_DIR / "models"
OUT_DIR_DS.mkdir(parents=True, exist_ok=True)
OUT_DIR_MD.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS_MAX = 5

# Parámetros de selección de variables (puedes ajustar)
FEAT_CUM_THRESHOLD = float(os.getenv("FEAT_CUM_THRESHOLD", 0.98))  # porcentaje acumulado a retener
FEAT_MIN_KEEP      = int(os.getenv("FEAT_MIN_KEEP", 3))            # mínimo de columnas a mantener
PI_N_REPEATS       = int(os.getenv("PI_N_REPEATS", 8))             # repeticiones para permutation importance

# ===================== Helpers =====================
def rmse(y_true, y_pred):
    return sqrt(((y_true - y_pred) ** 2).mean())

def label_observado(series):
    """1 = Observado ('O'). Si no hay 'O', fallback 1 = != 'R'."""
    s = series.fillna("DESCONOCIDO").astype(str).str.upper().str.strip()
    y = s.eq("O").astype(int)
    if y.nunique() < 2:
        y = s.ne("R").astype(int)
    return y

# Scorers
scorer_mae = make_scorer(mean_absolute_error, greater_is_better=False)
def scorer_rmse(est, X, y): return -rmse(y, est.predict(X))
scorer_r2  = make_scorer(r2_score)
def scorer_f1_pos(est, X, y): return f1_score(y, est.predict(X), average="binary", pos_label=1)
def scorer_recall_pos(est, X, y): return recall_score(y, est.predict(X), average="binary", pos_label=1)
def scorer_roc_auc(est, X, y):
    if hasattr(est, "predict_proba"): return roc_auc_score(y, est.predict_proba(X)[:,1])
    if hasattr(est, "decision_function"): return roc_auc_score(y, est.decision_function(X))
    return roc_auc_score(y, est.predict(X))

def build_preprocessor(num_cols, cat_cols):
    """Crea un ColumnTransformer con las columnas indicadas."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    # salida densa para compatibilidad
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, list(num_cols)),
        ("cat", cat_pipe, list(cat_cols)),
    ], remainder="drop", sparse_threshold=0.0)
    return pre

def _agg_importances_to_raw_cols(preprocessor, feat_names, importances, num_cols, cat_cols):
    """
    Agrega importancias de features transformados (incluye dummies)
    a nivel de columna original.
    """
    agg = {c: 0.0 for c in list(num_cols) + list(cat_cols)}
    for name, imp in zip(feat_names, importances):
        if name.startswith("num__"):
            raw = name.split("__", 1)[1]
            agg[raw] = agg.get(raw, 0.0) + float(imp)
        elif name.startswith("cat__"):
            tail = name.split("__", 1)[1]   # ej: "sexo_F"
            raw = tail.split("_", 1)[0]     # -> "sexo"
            agg[raw] = agg.get(raw, 0.0) + float(imp)
        else:
            raw = name
            if raw in agg:
                agg[raw] += float(imp)
    return agg

def select_columns_by_permutation(
    pipeline, X, y, num_cols, cat_cols, problem_type,
    seed=42, scoring_pref=None, n_repeats=8,
    keep_top_k=None, cum_threshold=0.98, min_keep=3
):
    """
    Calcula permutation importance sobre el pipeline (pre + modelo),
    agrega importancias a columnas originales y devuelve lista de columnas seleccionadas
    y un DataFrame con el ranking.
    """
    if scoring_pref is None:
        scoring_pref = "neg_mean_absolute_error" if problem_type == "reg" else "f1"
    scorer = get_scorer(scoring_pref)

    pi = permutation_importance(
        pipeline, X, y, scoring=scorer, n_repeats=n_repeats,
        random_state=seed, n_jobs=-1
    )
    pre = pipeline.named_steps["pre"]
    feat_names = pre.get_feature_names_out()
    agg = _agg_importances_to_raw_cols(pre, feat_names, pi.importances_mean, num_cols, cat_cols)

    # ranking por importancia agregada
    imp_items = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    total = sum(v for _, v in imp_items) or 1.0

    selected = []
    acum = 0.0
    for i, (col, val) in enumerate(imp_items, start=1):
        selected.append(col)
        acum += val / total
        if keep_top_k is not None and i >= keep_top_k:
            break
        if keep_top_k is None and acum >= cum_threshold and len(selected) >= min_keep:
            break

    rank_df = pd.DataFrame(imp_items, columns=["columna", "importancia_sumada"])
    return selected, rank_df

# ===================== Load =====================
print(f"Cargando dataset desde {DATA_PATH} ...")
df = pd.read_excel(DATA_PATH)

# Normaliza encabezados
df.columns = (df.columns.str.strip().str.lower()
              .str.replace(" ", "_")
              .str.replace("[^a-z0-9_]", "", regex=True))

print("Columnas detectadas:", list(df.columns))

TARGET_REG = "promedio_ponderado"
CLF_RAW    = "situacion_academica"

# Candidatas (usa solo las que existen y no están 100% vacías)
num_candidates = [
    "promedio_ultima_matricula",
    "anio_ciclo_est",
    "num_periodo_acad_matric",
    "anio_ingreso",
    "edad_en_ingreso",
]
cat_candidates = [
    "sexo",
    "facultad",
    "programa",
   # "permanencia",
    "ultimo_periodo_matriculado",
]

present = set(df.columns)
num_present = [c for c in num_candidates if c in present and df[c].notna().any()]
cat_present = [c for c in cat_candidates if c in present and df[c].notna().any()]

needed = num_present + [c for c in [TARGET_REG, CLF_RAW] if c in present]
df = df.dropna(subset=needed, how="any").reset_index(drop=True)

# Cast numéricos
for c in num_present + ([TARGET_REG] if TARGET_REG in df.columns else []):
    df[c] = pd.to_numeric(df[c], errors="coerce")

X = df[num_present + cat_present].copy()
y_reg = df[TARGET_REG]
y_clf = label_observado(df[CLF_RAW])

print("Numéricas:", num_present)
print("Categóricas:", cat_present)
print("Distribución y_clf:", y_clf.value_counts().to_dict())

# Ajuste de folds seguro para clasificación
counts = y_clf.value_counts()
min_per_class = counts.min() if counts.shape[0] >= 2 else 0
n_splits_clf = max(2, min(N_FOLDS_MAX, int(min_per_class))) if min_per_class > 0 else 0
if n_splits_clf == 0:
    warnings.warn("Clasificación omitida: no hay suficientes ejemplos de ambas clases.")

# ===================== Preprocess base (todas las columnas) =====================
pre_full = build_preprocessor(num_present, cat_present)

# ===================== Modelos =====================
# ——— REGRESIÓN ———
regressors = {
    "SVR_RBF": SVR(C=2.0, epsilon=0.2, kernel="rbf"),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1),
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=SEED),
    "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=600, random_state=SEED, n_jobs=-1),
    "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=7),
    "Ridge": Ridge(alpha=1.0, random_state=SEED),
    "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=SEED, max_iter=10000),
    "Lasso": Lasso(alpha=0.001, random_state=SEED, max_iter=10000),
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state=SEED),
}
if HAS_LGBM:
    regressors["LGBMRegressor"] = LGBMRegressor(
        n_estimators=700, learning_rate=0.05, random_state=SEED, n_jobs=-1
    )
else:
    print("⚠️ LightGBM no disponible: omitiendo LGBMRegressor.")
if HAS_CAT:
    regressors["CatBoostRegressor"] = CatBoostRegressor(
        iterations=1000, depth=6, learning_rate=0.05, random_seed=SEED,
        loss_function="MAE", verbose=False
    )
else:
    print("⚠️ CatBoost no disponible: omitiendo CatBoostRegressor.")

# ——— CLASIFICACIÓN ———
classifiers = {}
if n_splits_clf >= 2:
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED),
        "SVC_RBF": SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=7),
        "GaussianNB": GaussianNB(),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=SEED, class_weight="balanced"),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=600, random_state=SEED, class_weight="balanced", n_jobs=-1),
        "ExtraTreesClassifier": ExtraTreesClassifier(n_estimators=700, random_state=SEED, class_weight="balanced", n_jobs=-1),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=SEED),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(random_state=SEED),
        "AdaBoostClassifier": AdaBoostClassifier(random_state=SEED, n_estimators=500),
    }
    if HAS_LGBM:
        classifiers["LGBMClassifier"] = LGBMClassifier(
            n_estimators=800, learning_rate=0.05, random_state=SEED, class_weight="balanced", n_jobs=-1
        )
    else:
        print("⚠️ LightGBM no disponible: omitiendo LGBMClassifier.")
    if HAS_CAT:
        classifiers["CatBoostClassifier"] = CatBoostClassifier(
            iterations=1200, depth=6, learning_rate=0.05, random_seed=SEED,
            loss_function="Logloss", verbose=False, auto_class_weights="Balanced"
        )
    else:
        print("⚠️ CatBoost no disponible: omitiendo CatBoostClassifier.")

# ===================== Evaluación (con todas las columnas) =====================
kf_reg = KFold(n_splits=5, shuffle=True, random_state=SEED)
rows_reg, best_reg, best_reg_score = [], None, -np.inf
scorers_reg = {"MAE": scorer_mae, "RMSE": scorer_rmse, "R2": scorer_r2}

print("\n=== ENTRENANDO REGRESIÓN ===")
for name, model in regressors.items():
    pipe = Pipeline([("pre", pre_full), ("model", model)])
    cv = cross_validate(pipe, X, y_reg, cv=kf_reg, scoring=scorers_reg, n_jobs=-1)
    mae_mean, rmse_mean, r2_mean = -cv["test_MAE"].mean(), -cv["test_RMSE"].mean(), cv["test_R2"].mean()
    rows_reg.append({"modelo": name, "MAE": mae_mean, "RMSE": rmse_mean, "R2": r2_mean})
    # Elegimos por MAE más bajo
    if -mae_mean > best_reg_score:
        best_reg_score, best_reg = -mae_mean, (name, model)
df_reg = pd.DataFrame(rows_reg).sort_values(["MAE","RMSE"]).reset_index(drop=True)
df_reg.to_csv(OUT_DIR_DS / "metrics_regresion.csv", index=False)
print(df_reg)

rows_clf = []
best_clf, best_clf_score = None, -np.inf
if n_splits_clf >= 2 and len(classifiers) > 0:
    kf_clf = StratifiedKFold(n_splits=n_splits_clf, shuffle=True, random_state=SEED)
    scorers_clf = {"F1_pos": scorer_f1_pos, "Recall_pos": scorer_recall_pos, "ROC_AUC": scorer_roc_auc}
    print(f"\n=== ENTRENANDO CLASIFICACIÓN (n_splits={n_splits_clf}) ===")
    for name, model in classifiers.items():
        pipe = Pipeline([("pre", pre_full), ("model", model)])
        cv = cross_validate(pipe, X, y_clf, cv=kf_clf, scoring=scorers_clf, n_jobs=-1, error_score="raise")
        f1m, rec, roc = cv["test_F1_pos"].mean(), cv["test_Recall_pos"].mean(), cv["test_ROC_AUC"].mean()
        rows_clf.append({"modelo": name, "F1_pos": f1m, "Recall_pos": rec, "ROC_AUC": roc})
        if f1m > best_clf_score:
            best_clf_score, best_clf = f1m, (name, model)
    df_clf = pd.DataFrame(rows_clf).sort_values(["F1_pos","ROC_AUC"], ascending=[False, False]).reset_index(drop=True)
    df_clf.to_csv(OUT_DIR_DS / "metrics_clasificacion.csv", index=False)
    print(df_clf)
else:
    print("\n⚠️ Clasificación omitida por insuficiencia de clases.")

# ===================== Selección de variables + Export =====================
print("\n=== SELECCIÓN DE VARIABLES Y EXPORTACIÓN ===")

# ---------- REGRESIÓN ----------
if best_reg:
    reg_name, reg_model = best_reg
    print(f"\n[REG] Mejor modelo preliminar: {reg_name}. Seleccionando variables por permutation importance…")

    # 1) Pipeline preliminar con todas las columnas
    pre_full_reg = build_preprocessor(num_present, cat_present)
    pipe_full_reg = Pipeline([("pre", pre_full_reg), ("model", reg_model)])
    pipe_full_reg.fit(X, y_reg)

    # 2) Selección a nivel de columnas originales
    selected_reg_cols, rank_reg = select_columns_by_permutation(
        pipe_full_reg, X, y_reg,
        num_cols=num_present, cat_cols=cat_present,
        problem_type="reg", seed=SEED,
        scoring_pref="neg_mean_absolute_error",
        n_repeats=PI_N_REPEATS, cum_threshold=FEAT_CUM_THRESHOLD, min_keep=FEAT_MIN_KEEP
    )
    rank_reg.to_csv(OUT_DIR_DS / "feature_ranking_regresion.csv", index=False)

    # 3) Reconstruir preprocesador con columnas seleccionadas
    num_sel_reg = [c for c in selected_reg_cols if c in num_present]
    cat_sel_reg = [c for c in selected_reg_cols if c in cat_present]
    pre_sel_reg = build_preprocessor(num_sel_reg, cat_sel_reg)

    # 4) Revalidar CV con columnas seleccionadas
    pipe_sel_reg = Pipeline([("pre", pre_sel_reg), ("model", reg_model)])
    cv_sel = cross_validate(pipe_sel_reg, X[selected_reg_cols], y_reg, cv=kf_reg, scoring=scorers_reg, n_jobs=-1)
    mae_sel, rmse_sel, r2_sel = -cv_sel["test_MAE"].mean(), -cv_sel["test_RMSE"].mean(), cv_sel["test_R2"].mean()
    print(f"[REG] Con selección → MAE={mae_sel:.4f}, RMSE={rmse_sel:.4f}, R2={r2_sel:.4f}")

    # 5) Fit final y export
    pipe_sel_reg.fit(X[selected_reg_cols], y_reg)
    joblib.dump(pipe_sel_reg, OUT_DIR_MD / "best_regressor.joblib")

    meta_reg = {
        "model_name": reg_name,
        "selected_columns": selected_reg_cols,
        "selected_num": num_sel_reg,
        "selected_cat": cat_sel_reg,
        "cv_metrics": {"MAE": mae_sel, "RMSE": rmse_sel, "R2": r2_sel}
    }
    (OUT_DIR_MD / "used_features_reg.json").write_text(json.dumps(meta_reg, ensure_ascii=False, indent=2))
    print(f"[REG] {reg_name} guardado con {len(selected_reg_cols)} columnas como models/best_regressor.joblib")
    print(f"[REG] Variables usadas (JSON): {OUT_DIR_MD / 'used_features_reg.json'}")
else:
    print("⚠️ No se encontró mejor modelo de regresión para exportar.")

# ---------- CLASIFICACIÓN ----------
if best_clf:
    clf_name, clf_model = best_clf
    print(f"\n[CLF] Mejor modelo preliminar: {clf_name}. Seleccionando variables por permutation importance…")

    pre_full_clf = build_preprocessor(num_present, cat_present)
    pipe_full_clf = Pipeline([("pre", pre_full_clf), ("model", clf_model)])
    pipe_full_clf.fit(X, y_clf)

    selected_clf_cols, rank_clf = select_columns_by_permutation(
        pipe_full_clf, X, y_clf,
        num_cols=num_present, cat_cols=cat_present,
        problem_type="clf", seed=SEED,
        scoring_pref="f1", n_repeats=PI_N_REPEATS,
        cum_threshold=FEAT_CUM_THRESHOLD, min_keep=FEAT_MIN_KEEP
    )
    rank_clf.to_csv(OUT_DIR_DS / "feature_ranking_clasificacion.csv", index=False)

    num_sel_clf = [c for c in selected_clf_cols if c in num_present]
    cat_sel_clf = [c for c in selected_clf_cols if c in cat_present]
    pre_sel_clf = build_preprocessor(num_sel_clf, cat_sel_clf)

    kf_clf = StratifiedKFold(n_splits=n_splits_clf, shuffle=True, random_state=SEED)
    scorers_clf = {"F1_pos": scorer_f1_pos, "Recall_pos": scorer_recall_pos, "ROC_AUC": scorer_roc_auc}
    pipe_sel_clf = Pipeline([("pre", pre_sel_clf), ("model", clf_model)])
    cv_sel = cross_validate(pipe_sel_clf, X[selected_clf_cols], y_clf, cv=kf_clf, scoring=scorers_clf, n_jobs=-1)
    f1m, rec, roc = cv_sel["test_F1_pos"].mean(), cv_sel["test_Recall_pos"].mean(), cv_sel["test_ROC_AUC"].mean()
    print(f"[CLF] Con selección → F1_pos={f1m:.4f}, Recall_pos={rec:.4f}, ROC_AUC={roc:.4f}")

    pipe_sel_clf.fit(X[selected_clf_cols], y_clf)
    joblib.dump(pipe_sel_clf, OUT_DIR_MD / "best_classifier.joblib")

    meta_clf = {
        "model_name": clf_name,
        "selected_columns": selected_clf_cols,
        "selected_num": num_sel_clf,
        "selected_cat": cat_sel_clf,
        "cv_metrics": {"F1_pos": f1m, "Recall_pos": rec, "ROC_AUC": roc}
    }
    (OUT_DIR_MD / "used_features_clf.json").write_text(json.dumps(meta_clf, ensure_ascii=False, indent=2))
    print(f"[CLF] {clf_name} guardado con {len(selected_clf_cols)} columnas como models/best_classifier.joblib")
    print(f"[CLF] Variables usadas (JSON): {OUT_DIR_MD / 'used_features_clf.json'}")
else:
    print("⚠️ Clasificador no exportado (no hubo clases suficientes o no se seleccionó mejor modelo).")

print("\n✅ Listo.")
print(f"- Métricas regresión: {OUT_DIR_DS / 'metrics_regresion.csv'}")
print(f"- Métricas clasificación: {OUT_DIR_DS / 'metrics_clasificacion.csv'}")
print(f"- Rankings: {OUT_DIR_DS / 'feature_ranking_regresion.csv'} y {OUT_DIR_DS / 'feature_ranking_clasificacion.csv'}")
print(f"- Modelos/Metadatos: {OUT_DIR_MD}")
