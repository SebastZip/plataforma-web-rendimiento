import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def cargar_y_preparar_datos(ruta_excel='C:\\Users\\Sebas 2\\Desktop\\plataforma-web-rendimiento\\modelo-predictivo\\dataset\\Students_Performance_data_set.xlsx'):
    df = pd.read_excel(ruta_excel)

    columnas_a_eliminar = [
        'University Admission year',  
        'Program',  
        'What are the skills do you have ?',  
        'What is you interested area?'      
    ]
    df.drop(columns=columnas_a_eliminar, inplace=True)

    df['Average attendance on class'] = pd.to_numeric(df['Average attendance on class'], errors='coerce')
    df['What is your relationship status?'] = df['What is your relationship status?'].replace({'In a relationship': 'Relationship'})

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

    df = pd.get_dummies(df, columns=[
        'Status of your English language proficiency',
        'What is your relationship status?'
    ], drop_first=True)

    df.dropna(inplace=True)

    X = df.drop(columns=['What is your current CGPA?'])
    y = df['What is your current CGPA?']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n📊 Tipos de datos en X:")
    print(X.dtypes.value_counts()) 

    print("\n📈 Tamaños:")
    print("📁 X_train:", X_train.shape)
    print("📁 X_test:", X_test.shape)
    print("📁 y_train:", y_train.shape)
    print("📁 y_test:", y_test.shape)

    df.to_csv("C:\\Users\\Sebas 2\\Desktop\\plataforma-web-rendimiento\\modelo-predictivo\\dataset\\dataset_regresion.csv", index=False)

    # --- NUEVO: Análisis del target (CGPA) ---
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

    print("\n🧼 ¿Hay valores nulos en el dataset?")
    print(df.isnull().sum().sum(), "nulos en total")
    print("\n🔍 Nulos por columna:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    print("\n📋 Tipos de datos por columna:")
    print(df.dtypes.value_counts())

    # Listado específico por tipo
    bool_cols = df.select_dtypes(include='bool').columns.tolist()
    int_cols = df.select_dtypes(include='int').columns.tolist()
    float_cols = df.select_dtypes(include='float').columns.tolist()

    print("\n🔹 Columnas booleanas:", bool_cols)
    print("🔹 Columnas enteras:", int_cols)
    print("🔹 Columnas flotantes:", float_cols)

    print("\n📌 Filas completamente vacías:")
    print(df[df.isnull().all(axis=1)])

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    cargar_y_preparar_datos()
