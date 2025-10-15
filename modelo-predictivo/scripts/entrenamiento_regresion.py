# train_regression.py
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

from deap import base, creator, tools, algorithms
from preprocesamiento3 import preparar_regresion

# =========================
# 0) Configuración y paths
# =========================
OUTDIR = "fastapi-backend/modelos"
os.makedirs(OUTDIR, exist_ok=True)

# =========================
# 1) Carga de datos
# =========================
X_train, X_test, y_train, y_test, feature_names = preparar_regresion(
    ruta_excel="C:\\Users\\Sebas 2\\Desktop\\plataforma-web-rendimiento\\modelo-predictivo\\dataset\\datos_estudiantes_FISI_peru_sinteticos_1800.xlsx"
)
# Asegurar DataFrames con columnas en el mismo orden base
X_train = pd.DataFrame(X_train, columns=feature_names)
X_test  = pd.DataFrame(X_test,  columns=feature_names)

# RAYOS X: chequeos rápidos
print("=== RAYOS X INICIO ===")
print("X_train shape:", X_train.shape, "| X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape, "| y_test shape:", y_test.shape)
print("¿Columnas iguales y mismo orden? ->", list(X_train.columns) == list(X_test.columns))
print("NaNs por columna (TRAIN) TOP 10:\n", X_train.isna().sum().sort_values(ascending=False).head(10))
print("NaNs por columna (TEST)  TOP 10:\n", X_test.isna().sum().sort_values(ascending=False).head(10))
print("=== RAYOS X FIN ===\n")

# ===========================================================
# 2) GA: preprocesamiento CONSISTENTE (imputar + escalar TRAIN)
# ===========================================================
# Imputamos usando TODAS las columnas para que el GA vea una base limpia
ga_imputer = SimpleImputer(strategy="median").fit(X_train)
Xtr_imp_full = ga_imputer.transform(X_train)

ga_scaler = StandardScaler().fit(Xtr_imp_full)
Xtr_scaled_full = ga_scaler.transform(Xtr_imp_full)

# Fix re-ejecuciones en notebooks: borrar tipos previos
for name in ["FitnessMin", "Individual"]:
    if hasattr(creator, name):
        delattr(creator, name)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.randint, 0, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X_train.shape[1])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_mse_cv(individual):
    # Evitar individuo vacío
    if sum(individual) == 0:
        return (1e6,)
    idx = [i for i, bit in enumerate(individual) if bit == 1]
    X_sel = Xtr_scaled_full[:, idx]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mses = []
    yv = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)
    for tr, va in kf.split(X_sel):
        model = LinearRegression().fit(X_sel[tr], yv[tr])
        pred = model.predict(X_sel[va])
        mses.append(mean_squared_error(yv[va], pred))
    return (np.mean(mses),)

toolbox.register("evaluate", eval_mse_cv)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

pop = toolbox.population(n=40)
pop, _ = algorithms.eaSimple(pop, toolbox, cxpb=0.45, mutpb=0.25, ngen=20, verbose=False)
best = tools.selBest(pop, k=1)[0]
sel_idx = [i for i, bit in enumerate(best) if bit == 1]
sel_names = [feature_names[i] for i in sel_idx]
print("\n🧬 Features GA ({}):".format(len(sel_names)), sel_names)

# =====================================================
# 3) Ranking CatBoost -> TOP-12 (con datos imputados)
# =====================================================
# Imputamos SOLO las columnas elegidas por el GA para el ranking coherente
rank_imputer = SimpleImputer(strategy="median").fit(X_train[sel_names])
Xtr_ga_imp = pd.DataFrame(rank_imputer.transform(X_train[sel_names]), columns=sel_names, index=X_train.index)

cat_rank = CatBoostRegressor(verbose=0, random_state=42)
cat_rank.fit(Xtr_ga_imp, y_train)
imp = cat_rank.feature_importances_.astype(float)

# Importancias CatBoost sobre selección GA
imp_df = pd.DataFrame({"feature": sel_names, "catboost_importance": imp}) \
         .sort_values("catboost_importance", ascending=False).reset_index(drop=True)
imp_df["rank_catboost_ga"] = np.arange(1, len(imp_df) + 1)

print("\n📊 Importancias CatBoost (selección GA) — Top 10:")
print(imp_df.head(10))

# Guardar trazabilidad GA
imp_df.to_csv(os.path.join(OUTDIR, "importancias_catboost_sobre_GA.csv"), index=False)
pd.Series(sel_names, name="features_GA").to_csv(os.path.join(OUTDIR, "features_GA.csv"), index=False)

# Elegimos TOP-12
top_12 = imp_df["feature"].head(12).tolist()
print("🏆 TOP-12:", top_12)
pd.Series(top_12, name="top_12").to_csv(os.path.join(OUTDIR, "top_12.csv"), index=False)

# Plot importancias (Top-20)
topN = min(20, len(imp_df))
plt.figure(figsize=(10, 6))
plt.barh(imp_df["feature"][:topN][::-1], imp_df["catboost_importance"][:topN][::-1])
plt.title(f"CatBoost Feature Importance (selección GA) - Top {topN}")
plt.xlabel("Importance")
plt.tight_layout(); plt.show()

# ============================================================
# 4) Pipeline FINAL: imputar (TRAIN->TEST) y escalar con TOP-12
# ============================================================
imputer = SimpleImputer(strategy="median").fit(X_train[top_12])

Xtr_imputed = imputer.transform(X_train[top_12])
Xte_imputed = imputer.transform(X_test[top_12])

scaler = StandardScaler().fit(Xtr_imputed)
Xtr = scaler.transform(Xtr_imputed)
Xte = scaler.transform(Xte_imputed)

# RAYOS X: confirmar orden de columnas
print("\nOrden de columnas TOP-12 (para pipeline final):")
print(top_12)

# ========================
# 5) Entrenamiento modelos
# ========================
models = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoost": GradientBoostingRegressor(random_state=42),
    "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1),
    "LightGBM": LGBMRegressor(random_state=42),
    "CatBoost": CatBoostRegressor(verbose=0, random_state=42),
}

best_name, best_model, best_preds, best_r2 = None, None, None, -1
for name, mdl in models.items():
    mdl.fit(Xtr, y_train)
    pr = mdl.predict(Xte)
    rmse = mean_squared_error(y_test, pr, squared=False)
    mae = mean_absolute_error(y_test, pr)
    r2 = r2_score(y_test, pr)
    print(f"{name:12s}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    if r2 > best_r2:
        best_name, best_model, best_preds, best_r2 = name, mdl, pr, r2

print(f"\n✅ Mejor modelo TEST: {best_name} | R²={best_r2:.4f}")

# ===========================
# 6) Guardar artefactos claves
# ===========================
joblib.dump(best_model, os.path.join(OUTDIR, "modelo_entrenado2.pkl"))
joblib.dump(scaler,      os.path.join(OUTDIR, "scaler2.pkl"))
joblib.dump(imputer,     os.path.join(OUTDIR, "reg_imputer.pkl"))
joblib.dump(top_12,      os.path.join(OUTDIR, "top_12_features2.pkl"))

# ============
# 7) Graficado
# ============
cmp = pd.DataFrame({"Real": y_test.values, "Predicho": best_preds}).reset_index(drop=True)
print("\nMuestras TEST:\n", cmp.head(20))
plt.figure(figsize=(10, 6))
plt.plot(cmp["Real"][:100].values, label="Real")
plt.plot(cmp["Predicho"][:100].values, label="Predicho")
plt.title(f"Real vs Predicho ({best_name})")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# ==========================================================
# 8) Importancias del MEJOR MODELO (sobre pipeline final)
# ==========================================================
feat_order = top_12  # referencia explícita al orden
if hasattr(best_model, "feature_importances_"):
    fi = np.array(best_model.feature_importances_, dtype=float)
    fi_df = pd.DataFrame({"feature": feat_order, "importance_model": fi})
    metodo = "feature_importances_"
else:
    # Permutation Importance en TEST (robusto para cualquier modelo)
    # Usa R² como métrica para consistencia
    try:
        pi = permutation_importance(best_model, Xte, y_test.values, n_repeats=10, random_state=42, scoring="r2")
        fi_df = pd.DataFrame({
            "feature": feat_order,
            "importance_model": pi.importances_mean,
            "importance_std":    pi.importances_std
        })
        metodo = "permutation_importance (R²)"
    except Exception as e:
        print("⚠️ Error en permutation_importance:", e)
        # fallback: importancia por pérdida de RMSE (manual, simple)
        base_rmse = mean_squared_error(y_test, best_model.predict(Xte), squared=False)
        drops = []
        for j in range(Xte.shape[1]):
            Xte_perm = Xte.copy()
            rng = np.random.RandomState(42)
            rng.shuffle(Xte_perm[:, j])
            rmse_perm = mean_squared_error(y_test, best_model.predict(Xte_perm), squared=False)
            drops.append(rmse_perm - base_rmse)
        fi_df = pd.DataFrame({"feature": feat_order, "importance_model": np.array(drops)})
        metodo = "manual_permutation_RMSE_drop"

fi_df = fi_df.sort_values("importance_model", ascending=False).reset_index(drop=True)
fi_df["rank_model"] = np.arange(1, len(fi_df) + 1)

print(f"\n🏁 Importancias del mejor modelo ({best_name}) usando {metodo}:")
print(fi_df)

# Guardar CSV y gráfico
fi_df.to_csv(os.path.join(OUTDIR, "importancias_mejor_modelo_TEST.csv"), index=False)

plt.figure(figsize=(10, 6))
plt.barh(fi_df["feature"][::-1], fi_df["importance_model"][::-1])
plt.title(f"Importancias ({best_name}) en TEST · {metodo}")
plt.xlabel("Importance")
plt.tight_layout(); plt.show()
