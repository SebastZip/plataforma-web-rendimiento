import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def cargar_y_preparar_datos(ruta_excel='C:\\Users\\Sebas 2\\Desktop\\plataforma-web-rendimiento\\modelo-predictivo\\dataset\\Students_Performance_data_set.xlsx'):
    df = pd.read_excel(ruta_excel)

    # Normalizar CGPA y SGPA a base 20
    df['What is your current CGPA?'] *= 5
    df['What was your previous SGPA?'] *= 5

    # Eliminar columnas irrelevantes
    columnas_a_eliminar = [
        'University Admission year',  
        'Program',  
        'What are the skills do you have ?',  
        'What is you interested area?'      
    ]
    df.drop(columns=columnas_a_eliminar, inplace=True)

    # Limpiar campos
    df['Average attendance on class'] = pd.to_numeric(df['Average attendance on class'], errors='coerce')
    df['What is your relationship status?'] = df['What is your relationship status?'].replace({'In a relationship': 'Relationship'})

    # Mapear campos binarios
    binarias = [
        'Gender',
        'Do you have meritorious scholarship ?',
        'Do you use University transportation?',
        'Do you use smart phone?',
        'Do you have personal Computer?',
        'Did you ever fall in probation?',
        'Did you ever got suspension?',
        'Do you attend in teacher consultancy for any kind of academical problems?',
        'Are you engaged with any co-curriculum activities?',
        'With whom you are living with?',
        'Do you have any health issues?',
        'Do you have any physical disabilities?',
        'What is your preferable learning mode?'
    ]

    for col in binarias:
        df[col] = df[col].map({
            'Yes': 1, 'No': 0, 'N': 0, 
            'Male': 1, 'Female': 0, 
            'Family': 0, 'Bachelor': 1, 
            'Offline': 0, 'Online': 1
        })

    # Dummies
    df = pd.get_dummies(df, columns=[
        'Status of your English language proficiency',
        'What is your relationship status?'
    ], drop_first=True)

    # --- Nuevas variables ---
    df['Creditos_Normalizados'] = df['How many Credit did you have completed?'] * (226 / df['How many Credit did you have completed?'].max())

    current_year = 2024
    df['Semestres_Reales'] = (current_year - df['University Admission year']) * 2
    df['Semestres_Teoricos'] = np.ceil((df['Creditos_Normalizados'] / 226) * 10).astype(int)
    df['Semestres_Extra'] = df['Semestres_Reales'] - df['Semestres_Teoricos']

    # Eliminar nulos
    df.dropna(inplace=True)

    # Variables finales
    X = df[['Creditos_Normalizados', 'Semestres_Reales']]
    y = df['What is your current CGPA?']

    # División
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Guardar
    df.to_csv("C:\\Users\\Sebas 2\\Desktop\\plataforma-web-rendimiento\\modelo-predictivo\\dataset\\dataset_regresion.csv", index=False, float_format='%.3f')

    # Reportes
    print("\n📊 Estadísticas del CGPA (target):")
    print(y.describe())

    plt.figure(figsize=(8, 4))
    sns.histplot(y, bins=20, kde=True)
    plt.title("Distribución del CGPA")
    plt.xlabel("CGPA")
    plt.ylabel("Frecuencia")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n🧼 Nulos totales:", df.isnull().sum().sum())
    print("\n🔍 Nulos por columna:\n", df.isnull().sum()[df.isnull().sum() > 0])

    print("\n📋 Tipos de datos:")
    print(df.dtypes.value_counts())

    # Evaluación técnica
    print("\n📊 Análisis técnico de creditaje y semestres:")

    print("\n🔹 Créditos normalizados:")
    print("Mínimo:", df['Creditos_Normalizados'].min())
    print("Máximo:", df['Creditos_Normalizados'].max())
    print("Media:", df['Creditos_Normalizados'].mean())
    print("Valores fuera del rango típico (0 - 260):")
    print(df[df['Creditos_Normalizados'] > 260][['Creditos_Normalizados']])

    print("\n🔹 Semestres reales:")
    print("Mínimo:", df['Semestres_Reales'].min())
    print("Máximo:", df['Semestres_Reales'].max())
    print("Media:", df['Semestres_Reales'].mean())
    print("¿Hay estudiantes con más de 14 semestres?")
    print(df[df['Semestres_Reales'] > 14][['Semestres_Reales']])

    print("\n🔹 Semestres extra (atrasos):")
    print("Mínimo:", df['Semestres_Extra'].min())
    print("Máximo:", df['Semestres_Extra'].max())
    print("Media:", df['Semestres_Extra'].mean())
    print("Estudiantes con más de 4 semestres extra:")
    print(df[df['Semestres_Extra'] > 4][['Semestres_Reales', 'Semestres_Teoricos', 'Semestres_Extra']])

    print("\n✅ Evaluación técnica completada.")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    cargar_y_preparar_datos()
