# train_regression.py
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from deap import base, creator, tools, algorithms
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from preprocesamiento3  import cargar_y_preparar_datos

# 1) Obtener splits limpios desde el preprocesamiento
X_train, X_test, y_train, y_test, feature_names = cargar_y_preparar_datos(
    ruta_excel="C:\\Users\\Sebas 2\\Desktop\\plataforma-web-rendimiento\\modelo-predictivo\\dataset\\datos_estudiantes_FISI_peru_sinteticos_1800.xlsx"
)

# 2) GA para selección de features en TRAIN (sin tocar TEST)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.randint, 0, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X_train.shape[1])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Escalado para la eval interna del GA
scaler_all = StandardScaler().fit(X_train.values)
Xtr_scaled_all = scaler_all.transform(X_train.values)

def eval_mse_cv(individual):
    if sum(individual) == 0:
        return (1e6,)
    idx = [i for i, bit in enumerate(individual) if bit == 1]
    X_sel = Xtr_scaled_all[:, idx]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mses = []
    for tr, va in kf.split(X_sel):
        model = LinearRegression().fit(X_sel[tr], y_train.values[tr])
        pred = model.predict(X_sel[va])
        mses.append(mean_squared_error(y_train.values[va], pred))
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
print("\n🧬 Features GA:", sel_names)

# 3) CatBoost para ranking e ir a TOP-12
cat = CatBoostRegressor(verbose=0, random_state=42)
cat.fit(X_train[sel_names], y_train)
imp = cat.feature_importances_
top_12 = [n for n, _ in sorted(zip(sel_names, imp), key=lambda x: x[1], reverse=True)[:12]]
print("🏆 TOP-12:", top_12)

# 4) Escalador FINAL para los TOP-12 (fit en TRAIN, transform en TEST)
scaler = StandardScaler().fit(X_train[top_12])
Xtr = scaler.transform(X_train[top_12])
Xte = scaler.transform(X_test[top_12])

# 5) Modelos y evaluación en TEST
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
    print(f"{name:12s}  RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.3f}")
    if r2 > best_r2:
        best_name, best_model, best_preds, best_r2 = name, mdl, pr, r2

print(f"\n✅ Mejor modelo TEST: {best_name} | R²={best_r2:.3f}")

# 6) Guardar artefactos (para FastAPI)
joblib.dump(best_model, "fastapi-backend/modelos/modelo_entrenado2.pkl")
joblib.dump(scaler, "fastapi-backend/modelos/scaler2.pkl")
joblib.dump(top_12, "fastapi-backend/modelos/top_12_features2.pkl")

# 7) Gráfico real vs predicho (TEST)
cmp = pd.DataFrame({"Real": y_test.values, "Predicho": best_preds}).reset_index(drop=True)
print("\nMuestras TEST:\n", cmp.head(20))
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(cmp["Real"][:100].values, label="Real")
plt.plot(cmp["Predicho"][:100].values, label="Predicho")
plt.title(f"Real vs Predicho ({best_name})")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
