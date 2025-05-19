# import pandas as pd
# from sklearn.model_selection import train_test_split

# def cargar_y_preparar_datos():
#     # Cargar el dataset
#     ruta_csv = 'C:\\Users\\Sebas 2\\Desktop\\plataforma-web-rendimiento\\modelo-predictivo\\dataset\\estudiantes_limpio.csv'
#     df = pd.read_csv(ruta_csv)

#     # Agrupamiento de GRADE
#     def agrupar_grado(grade):
#         if grade in [0, 1, 2]:
#             return 0  # Bajo rendimiento
#         elif grade in [3, 4, 5]:
#             return 1  # Rendimiento medio
#         else:
#             return 2  # Alto rendimiento

#     df['GRADE'] = df['GRADE'].apply(agrupar_grado)

 

#     # Separar características y etiquetas
#     X = df.drop(columns=['GRADE'])
#     y = df['GRADE']

#     # Separar en entrenamiento y prueba
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#     from imblearn.over_sampling import SMOTE

#     # Aplica SMOTE después del split
#     smote = SMOTE(random_state=42)
#     X_train, y_train = smote.fit_resample(X_train, y_train)

#     return X_train, X_test, y_train, y_test

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def cargar_y_preparar_datos(ruta_excel='C:\\Users\\Sebas 2\\Desktop\\plataforma-web-rendimiento\\modelo-predictivo\\dataset\\Students_Performance_data_set.xlsx'):
    # 1. Cargar el dataset
    df = pd.read_excel(ruta_excel)

    # 2. Crear variable objetivo (label)
    df['performance'] = np.where(
        df['What is your current CGPA?'] > df['What was your previous SGPA?'], 'Mejoró',
        np.where(df['What is your current CGPA?'] < df['What was your previous SGPA?'], 'Empeoró', 'Igual')
    )

    columnas_a_eliminar = [
        'University Admission year',  # todos son de años similares
        'Program',  # todos tienen el mismo valor
        'What is your current CGPA?',  # usada para crear la variable target
        'What are the skills do you have ?',  # demasiadas categorías únicas
        'What is you interested area?'       # muy dispersa
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
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'N': 0, 'Male': 1, 'Female': 0, 'Family': 0, 'Bachelor': 1, 'Offline': 0, 'Online': 1})

    df = pd.get_dummies(df, columns=[
    'Status of your English language proficiency',
    'What is your relationship status?'
    ], drop_first=True)

    mapa_target = {'Empeoró': 0, 'Igual': 1, 'Mejoró': 2}
    df['performance'] = df['performance'].map(mapa_target)

    X = df.drop(columns='performance')
    y = df['performance']

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("\n📊 Verificando tipos de datos en X:")
    print(X.dtypes.value_counts()) 

    print("\n🧼 Verificando valores nulos en X:")
    print(X.isnull().sum().sum())  # Debería dar 0

    print("\n📈 Distribución de clases en Y (entrenamiento RESAMPLEADO):")
    print(y_train_res.value_counts(normalize=True).round(3))

    print("📁 Tamaño de X_train_res:", X_train_res.shape)
    print("📁 Tamaño de X_test:", X_test.shape)

    print("\n🔎 Nulos por columna en X:")
    print(X.isnull().sum()[X.isnull().sum() > 0])

    # Mostrar las filas donde hay nulos en esas dos columnas específicas
    nulos_df = df[df['Average attendance on class'].isnull() | df['Do you have any health issues?'].isnull()]
    print("\n📌 Filas con valores nulos:")
    print(nulos_df[['Average attendance on class', 'Do you have any health issues?']])

    print(df.dtypes.value_counts())
    bool_cols = df.select_dtypes(include='bool').columns
    print("Variables booleanas reales:", bool_cols.tolist())
    int_cols = df.select_dtypes(include='int').columns
    print("Variables enteras reales:", int_cols.tolist())
    float_cols = df.select_dtypes(include='float').columns
    print("Variables float reales:", float_cols.tolist())
    return X_train_res, X_test, y_train_res, y_test

    