# train_clasificacion.py
# -*- coding: utf-8 -*-
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Evita bloqueo de plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    f1_score, classification_report, confusion_matrix
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer             # <-- NUEVO

from deap import base, creator, tools, algorithms

# ---------------- CONFIG ----------------
RUTA_EXCEL = "C:\\Users\\Sebas 2\\Desktop\\plataforma-web-rendimiento\\modelo-predictivo\\dataset\\datos_estudiantes_FISI_peru_sinteticos_1800.xlsx"

TARGET = "planea_matricularse_prox_ciclo"   # HE02
CLASE_POSITIVA = "No"                       # 1 = No continuará (riesgo)
ARTEFACTOS_PREFIX = "continuidad"           # nombres de archivos

TOP_N = 10
OUT_DIR = Path("fastapi-backend/modelos")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- PREPRO ----------------
def _load_and_clean(ruta_excel):
    df = pd.read_excel(ruta_excel)

    drop_cols = ["codigo_anonimo", "motivo_no_matricula", "escuela_profesional"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    bin_map = {"Sí": 1, "Si": 1, "sí": 1, "si": 1, "SÍ": 1, "No": 0, "NO": 0, "no": 0}
    for col in ["estado_observado", "desaprobo_alguna_asignatura",
                "beca_subvencion_economica", "planea_matricularse_prox_ciclo"]:
        if col in df.columns:
            df[col] = df[col].map(bin_map)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    return df

def preparar_clasificacion(ruta_excel, target_cls, clase_positiva_es="No",
                           test_size=0.2, random_state=42):
    df = _load_and_clean(ruta_excel)
    if target_cls not in df.columns:
        raise KeyError(f"Target '{target_cls}' no está en el dataset.")
    y_raw = df[target_cls].astype(int)
    # 1 = No continúa (riesgo) si la clase positiva es "No"
    y = (y_raw == 0).astype(int) if clase_positiva_es.lower().startswith("no") else y_raw
    X = df.drop(columns=[target_cls])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, X.columns.tolist()

# ---------------- 1) Datos ----------------
X_train, X_test, y_train, y_test, feature_names = preparar_clasificacion(
    ruta_excel=RUTA_EXCEL,
    target_cls=TARGET,
    clase_positiva_es=CLASE_POSITIVA
)

# ---------------- 2) GA: selección de features ----------------
creator.create("FitnessMax", base.Fitness, weights=(+1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.randint, 0, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X_train.shape[1])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

scaler_all = StandardScaler().fit(X_train.values)
Xtr_scaled_all = scaler_all.transform(X_train.values)

def eval_aucpr_cv(individual):
    if sum(individual) == 0:
        return (0.0,)
    idx = [i for i, bit in enumerate(individual) if bit == 1]
    X_sel = Xtr_scaled_all[:, idx]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    auc_prs = []
    for tr, va in kf.split(X_sel):
        lr = LogisticRegression(max_iter=400, class_weight="balanced")
        lr.fit(X_sel[tr], y_train.values[tr])
        probs = lr.predict_proba(X_sel[va])[:, 1]
        auc_prs.append(average_precision_score(y_train.values[va], probs))
    return (float(np.mean(auc_prs)),)

toolbox.register("evaluate", eval_aucpr_cv)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

pop = toolbox.population(n=40)
pop, _ = algorithms.eaSimple(pop, toolbox, cxpb=0.45, mutpb=0.25, ngen=18, verbose=False)

best = tools.selBest(pop, k=1)[0]
sel_idx = [i for i, bit in enumerate(best) if bit == 1]
sel_names = [feature_names[i] for i in sel_idx]
print("\n🧬 Features GA (clasificación):", sel_names)

# ---------------- 3) Ranking + TOP-N ----------------
rf_tmp = RandomForestClassifier(n_estimators=600, random_state=42, class_weight="balanced")
rf_tmp.fit(X_train[sel_names], y_train)
imp = rf_tmp.feature_importances_
top_features = [n for n, _ in sorted(zip(sel_names, imp), key=lambda x: x[1], reverse=True)[:TOP_N]]
print(f"🏆 TOP-{TOP_N}:", top_features)

# ---------------- 4) Imputador + Escalado FINAL ----------------
imputer = SimpleImputer(strategy="median").fit(X_train[top_features])    # <-- NUEVO
Xtr_imp = imputer.transform(X_train[top_features])                       # <-- NUEVO
Xte_imp = imputer.transform(X_test[top_features])                        # <-- NUEVO

scaler = StandardScaler().fit(Xtr_imp)
Xtr = scaler.transform(Xtr_imp)
Xte = scaler.transform(Xte_imp)

# ---------------- 5) Modelos ----------------
models = {
    "LogReg": LogisticRegression(max_iter=600, class_weight="balanced"),
    "RF": RandomForestClassifier(n_estimators=800, random_state=42, class_weight="balanced"),
    "GB": GradientBoostingClassifier(random_state=42),
    "SVM-RBF": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    "LGBM": LGBMClassifier(random_state=42, class_weight="balanced"),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42, scale_pos_weight=1.0),
}

best_name, best_model, best_probs, best_aucpr = None, None, None, -1
for name, mdl in models.items():
    mdl.fit(Xtr, y_train)
    probs = mdl.predict_proba(Xte)[:, 1]
    auc_pr  = average_precision_score(y_test, probs)
    auc_roc = roc_auc_score(y_test, probs)
    f1_05   = f1_score(y_test, (probs >= 0.5).astype(int))
    print(f"{name:9s}  AUC-PR={auc_pr:.3f}  AUC-ROC={auc_roc:.3f}  F1@0.5={f1_05:.3f}")
    if auc_pr > best_aucpr:
        best_name, best_model, best_probs, best_aucpr = name, mdl, probs, auc_pr

print(f"\n✅ Mejor modelo TEST (por AUC-PR): {best_name} | AUC-PR={best_aucpr:.3f}")

prec, rec, thr = precision_recall_curve(y_test, best_probs)
f1s = 2 * (prec * rec) / (prec + rec + 1e-9)
idx_best = int(np.nanargmax(f1s))
thr_best = float(thr[idx_best]) if idx_best < len(thr) else 0.5

y_pred = (best_probs >= thr_best).astype(int)
print(f"\n⭐ Umbral óptimo por F1: {thr_best:.3f}  | F1={f1s[idx_best]:.3f}  | Prec={prec[idx_best]:.3f}  | Rec={rec[idx_best]:.3f}")

print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred, digits=3))

plt.figure(figsize=(6,5))
plt.plot(rec, prec, label=f"{best_name}")
plt.scatter(rec[idx_best], prec[idx_best], s=60, label="Umbral óptimo")
plt.title(f"Curva Precisión-Recall | {best_name}")
plt.xlabel("Recall"); plt.ylabel("Precisión"); plt.grid(True); plt.legend()
plt.tight_layout(); plt.savefig(f"{ARTEFACTOS_PREFIX}_pr_curve.png", dpi=150)

# ---------------- 6) Guardar artefactos ----------------
joblib.dump(best_model,   OUT_DIR / f"{ARTEFACTOS_PREFIX}_modelo.pkl")
joblib.dump(imputer,      OUT_DIR / f"{ARTEFACTOS_PREFIX}_imputer.pkl")   # <-- NUEVO
joblib.dump(scaler,       OUT_DIR / f"{ARTEFACTOS_PREFIX}_scaler.pkl")
joblib.dump(top_features, OUT_DIR / f"{ARTEFACTOS_PREFIX}_features.pkl")
joblib.dump(thr_best,     OUT_DIR / f"{ARTEFACTOS_PREFIX}_umbral.pkl")
