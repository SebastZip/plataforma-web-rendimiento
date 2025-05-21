# from preprocesamiento import cargar_y_preparar_datos
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import pandas as pd

# # Cargar los datos
# X_train, X_test, y_train, y_test = cargar_y_preparar_datos()

# # Lista de modelos con parámetros razonables
# modelos = {
#     "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
#     "SVM": SVC(kernel='rbf', C=1, gamma='scale', random_state=42),
#     "KNN": KNeighborsClassifier(n_neighbors=5, weights='uniform'),
#     "Naive Bayes": GaussianNB(),
#     "Árbol de Decisión": DecisionTreeClassifier(max_depth=10, random_state=42)
# }

# # Resultados
# resultados = []

# for nombre, modelo in modelos.items():
#     modelo.fit(X_train, y_train)
#     y_pred = modelo.predict(X_test)
    
#     resultados.append({
#         "Modelo": nombre,
#         "Accuracy": accuracy_score(y_test, y_pred),
#         "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
#         "Recall": recall_score(y_test, y_pred, average='weighted'),
#         "F1-Score": f1_score(y_test, y_pred, average='weighted')
#     })

# # Mostrar resultados como tabla ordenada
# df_resultados = pd.DataFrame(resultados).sort_values(by='F1-Score', ascending=False)
# print("🔍 Comparación de modelos:\n")
# print(df_resultados.to_string(index=False))

import pandas as pd
from preprocesamiento import cargar_y_preparar_datos
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
ruta_modelo = 'modelo-predictivo/modelos/xgboost_model.joblib'
ruta_scaler = 'modelo-predictivo/modelos/xgboost_scaler.joblib'

# 1. Cargar los datos ya preprocesados y balanceados
X_train, X_test, y_train, y_test = cargar_y_preparar_datos()

# 2. Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Definir modelos con hiperparámetros optimizados
modelos = {
    'XGBoost': XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        subsample=1,
        colsample_bytree=1,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    ),
    'Decision Tree': DecisionTreeClassifier(
        criterion='entropy',
        class_weight='balanced',
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
}

# 4. Evaluar modelos
resultados = []

for nombre, modelo in modelos.items():
    modelo.fit(X_train_scaled, y_train)
    y_pred = modelo.predict(X_test_scaled)

    resultados.append({
        'Modelo': nombre,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred, average='weighted')
    })

# 5. Mostrar resultados
df_resultados = pd.DataFrame(resultados).sort_values(by='F1-Score', ascending=False)
print("\n📊 Comparación de modelos:\n")
print(df_resultados.to_string(index=False))

# 6. Matriz de confusión del mejor modelo
mejor_modelo = df_resultados.iloc[0]['Modelo']
modelo_final = modelos[mejor_modelo]
y_pred_final = modelo_final.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred_final, labels=[2, 1, 0])

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Mejoró', 'Igual', 'Empeoró'],
            yticklabels=['Mejoró', 'Igual', 'Empeoró'])
plt.title(f'Matriz de Confusión - {mejor_modelo}')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.tight_layout()
plt.show()

joblib.dump(modelo_final, ruta_modelo)
joblib.dump(scaler, ruta_scaler)

print("✅ Modelo y scaler exportados correctamente.")