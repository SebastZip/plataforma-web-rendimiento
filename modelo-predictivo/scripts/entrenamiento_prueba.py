# scripts/entrenar_modelos_excel.py
# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
from math import sqrt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score, roc_auc_score, f1_score, recall_score

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
import warnings

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
    "permanencia",
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

# ===================== Preprocess =====================
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
# salida densa para compatibilidad (HGB, CatBoost/LGBM ok)
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
pre = ColumnTransformer([
    ("num", num_pipe, num_present),
    ("cat", cat_pipe, cat_present),
], remainder="drop", sparse_threshold=0.0)

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

# ===================== Evaluación =====================
kf_reg = KFold(n_splits=5, shuffle=True, random_state=SEED)
rows_reg, best_reg, best_reg_score = [], None, -np.inf
scorers_reg = {"MAE": scorer_mae, "RMSE": scorer_rmse, "R2": scorer_r2}

print("\n=== ENTRENANDO REGRESIÓN ===")
for name, model in regressors.items():
    pipe = Pipeline([("pre", pre), ("model", model)])
    cv = cross_validate(pipe, X, y_reg, cv=kf_reg, scoring=scorers_reg, n_jobs=-1)
    mae_mean, rmse_mean, r2_mean = -cv["test_MAE"].mean(), -cv["test_RMSE"].mean(), cv["test_R2"].mean()
    rows_reg.append({"modelo": name, "MAE": mae_mean, "RMSE": rmse_mean, "R2": r2_mean})
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
        pipe = Pipeline([("pre", pre), ("model", model)])
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

# ===================== Guardado =====================
print("\n=== GUARDANDO MEJORES MODELOS ===")
if best_reg:
    name, model = best_reg
    pipe_reg = Pipeline([("pre", pre), ("model", model)])
    pipe_reg.fit(X, y_reg)
    joblib.dump(pipe_reg, OUT_DIR_MD / "best_regressor.joblib")
    print(f"[REG] {name} guardado como models/best_regressor.joblib")

if best_clf:
    name, model = best_clf
    pipe_clf = Pipeline([("pre", pre), ("model", model)])
    pipe_clf.fit(X, y_clf)
    joblib.dump(pipe_clf, OUT_DIR_MD / "best_classifier.joblib")
    print(f"[CLF] {name} guardado como models/best_classifier.joblib")

print("\n✅ Listo.")
print(f"- Métricas regresión: {OUT_DIR_DS / 'metrics_regresion.csv'}")
print(f"- Métricas clasificación: {OUT_DIR_DS / 'metrics_clasificacion.csv'}")
print(f"- Modelos: {OUT_DIR_MD}")
