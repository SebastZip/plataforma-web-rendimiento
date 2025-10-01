import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from deap import base, creator, tools, algorithms

# 1. Cargar dataset
df = pd.read_csv("modelo-predictivo\dataset\dataset_regresion.csv")

# 2. Preprocesamiento
target = 'What is your current CGPA?'
X = df.drop(columns=[target])
y = df[target]
X = X.applymap(lambda x: int(x) if isinstance(x, bool) else x)

# Guardamos todas las columnas originales antes de escalar
X_original = X.copy()

# 3. Configurar DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.randint, 0, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Escalamos temporalmente todo para usar DEAP (solo para esta parte)
X_scaled_all = StandardScaler().fit_transform(X_original)

def eval_mse(individual):
    if sum(individual) == 0:
        return (1e6,)
    selected_indices = [i for i, bit in enumerate(individual) if bit == 1]
    X_sel = X_scaled_all[:, selected_indices]
    model = LinearRegression()
    model.fit(X_sel, y)
    preds = model.predict(X_sel)
    mse = ((y - preds) ** 2).mean()
    return (mse,)

toolbox.register("evaluate", eval_mse)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# 4. Ejecutar algoritmo genético
population = toolbox.population(n=30)
result_pop, _ = algorithms.eaSimple(population, toolbox, cxpb=0.4, mutpb=0.2, ngen=15, verbose=False)

# 5. Obtener mejores características
top_ind = tools.selBest(result_pop, k=1)[0]
selected_features = [i for i, bit in enumerate(top_ind) if bit == 1]
selected_names = X.columns[selected_features]
print(f"\n🧬 Mejores características seleccionadas ({len(selected_names)}):\n{list(selected_names)}\n")

# 6. Importancia con CatBoost
X_sel_full = X_original[selected_names]
catboost = CatBoostRegressor(verbose=0, random_state=42)
catboost.fit(X_sel_full, y)
importances = catboost.feature_importances_
feature_importance_sorted = sorted(zip(selected_names, importances), key=lambda x: x[1], reverse=True)

print("📌 Importancia de cada característica seleccionada (CatBoost):\n")
for name, score in feature_importance_sorted:
    print(f"{name:60s} -> {score:.4f}")

# 7. Seleccionar top 12 características
top_12_features = [name for name, _ in feature_importance_sorted[:12]]

# 8. Escalar solo las top 12
X_top12 = X_original[top_12_features]
scaler = StandardScaler()
X_top12_scaled = scaler.fit_transform(X_top12)

# 9. Evaluar varios modelos
models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "SVR (RBF kernel)": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    "LightGBM Regressor": LGBMRegressor(random_state=42),
    "CatBoost Regressor": CatBoostRegressor(verbose=0, random_state=42),
}

results = {}
print("\n📊 Resultados de modelos con Top 12 características:\n")
for name, model in models.items():
    model.fit(X_top12_scaled, y)
    preds = model.predict(X_top12_scaled)
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)
    results[name] = {'model': model, 'mse': mse, 'r2': r2, 'preds': preds}
    print(f"🔹 {name}")
    print(f"   MSE: {mse:.4f}")
    print(f"   R² : {r2:.4f}\n")

# 10. Seleccionar el mejor modelo (mayor R²)
best_model_name = max(results, key=lambda x: results[x]['r2'])
best_model = results[best_model_name]['model']
best_preds = results[best_model_name]['preds']

print(f"✅ Mejor modelo: {best_model_name} con R² = {results[best_model_name]['r2']:.4f}")

# 11. Guardar el mejor modelo, scaler y top features
joblib.dump(best_model, 'fastapi-backend/modelos/modelo_entrenado.pkl')
joblib.dump(scaler, 'fastapi-backend/modelos/scaler.pkl')
joblib.dump(top_12_features, 'fastapi-backend/modelos/top_12_features.pkl')   

# 12. Mostrar y graficar comparación
comparison_df = pd.DataFrame({
    "Actual": y.values,
    "Predicted": best_preds
})
print("📈 Comparación de valores reales vs predichos (primeros 20):\n")
print(comparison_df.head(50))

# Gráfico
plt.figure(figsize=(10, 6))
plt.plot(y.values[:100], label='Actual', marker='o', linestyle='-')
plt.plot(best_preds[:100], label='Predicted', marker='x', linestyle='--')
plt.title(f"Comparación entre valores reales y predichos ({best_model_name})")
plt.xlabel("Índice de muestra")
plt.ylabel("CGPA")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
