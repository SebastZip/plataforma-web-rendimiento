#Paso1: Instalación de librerías
#pip install pandas numpy scikit-learn xgboost catboost geneticalgorithm2 imbalanced-learn joblib

# Paso 2: Importación de librerías
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from geneticalgorithm2 import geneticalgorithm2 as ga
from imblearn.over_sampling import SMOTE
import joblib

# Paso 3: Cargar dataset
df = pd.read_csv('../dataset/dataset_preprocesado.csv')

# Paso 4: Separar variables
X = df.drop('performance', axis=1)
y = df['performance']

# Paso 5: División inicial de entrenamiento y prueba
X_train_full, X_test_full, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Paso 6: Escalar para la función objetivo del GA
scaler_ga = StandardScaler()
X_train_full_scaled = scaler_ga.fit_transform(X_train_full)
X_test_full_scaled = scaler_ga.transform(X_test_full)

X_train_full_scaled = pd.DataFrame(X_train_full_scaled, columns=X_train_full.columns)
X_test_full_scaled = pd.DataFrame(X_test_full_scaled, columns=X_test_full.columns)

# Paso 7: Función objetivo para el Algoritmo Genético
def objective_function(variables):
    selected_indices = np.where(variables == 1)[0]
    if len(selected_indices) == 0:
        return 1.0  # Penalización por no seleccionar nada
    selected_features = X_train_full_scaled.columns[selected_indices]
    
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_full_scaled[selected_features], y_train)
    y_pred = model.predict(X_test_full_scaled[selected_features])
    acc = accuracy_score(y_test, y_pred)
    return 1.0 - acc  # Minimizar el error

# Paso 8: Configuración del algoritmo genético
varbound = np.array([[0, 1]] * X.shape[1], dtype=int)

model = ga(
    function=objective_function,
    dimension=X.shape[1],
    variable_type='int',
    variable_boundaries=varbound,
    algorithm_parameters={
        'max_num_iteration': 300,
        'population_size': 100,
        'elit_ratio': 0.05,
        'parents_portion': 0.3,
        'crossover_type': 'uniform',
        'mutation_type': 'uniform_by_x',
        'selection_type': 'roulette',
        'max_iteration_without_improv': 100
    }
)

result = model.run()

# Paso 9: Extraer las mejores columnas
best_variables = result['variable']
selected_indices = np.where(best_variables == 1)[0]
selected_columns = X.columns[selected_indices]
print("\n✅ Características seleccionadas:")
print(list(selected_columns))

# Paso 10: Preprocesamiento y SMOTE
X_selected = X[selected_columns]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Balanceo con SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Paso 11: Definir y entrenar modelos
modelos = {
    "logistic": LogisticRegression(max_iter=2000, class_weight='balanced'),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "xgboost": XGBClassifier(eval_metric="mlogloss", random_state=42),
    "catboost": CatBoostClassifier(verbose=0, random_state=42)
}

resultados = []

for nombre, modelo in modelos.items():
    modelo.fit(X_train_resampled, y_train_resampled)
    y_pred = modelo.predict(X_test)
    print(f"\n== {nombre.upper()} ==")
    print(classification_report(y_test, y_pred))
    resultados.append({
        'Modelo': nombre,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred, average='weighted')
    })

# Paso 12: Mostrar resultados
df_resultados = pd.DataFrame(resultados).sort_values(by='F1-Score', ascending=False)
print("\n📊 Comparación de modelos:\n")
print(df_resultados.to_string(index=False))

# Paso 13: Guardar el mejor modelo
mejor_modelo_nombre = df_resultados.iloc[0]['Modelo']
mejor_modelo = modelos[mejor_modelo_nombre]

# Ruta donde se guardará el modelo
ruta_modelo = f"../modelos/{mejor_modelo_nombre}_modelo.pkl"
joblib.dump(mejor_modelo, ruta_modelo)

# Guardar también el scaler utilizado para transformar los datos
joblib.dump(scaler, "../modelos/escalador.pkl")

# Guardar las columnas seleccionadas
joblib.dump(list(selected_columns), "../modelos/columnas_seleccionadas.pkl")

print(f"\n💾 Modelo guardado como: {ruta_modelo}")
