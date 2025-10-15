# scripts/analizar_importancias.py
# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance

# Modelos
from sklearn.ensemble import RandomForestRegressor

# Opcionales
HAS_CAT = True
try:
    from catboost import CatBoostRegressor
except Exception:
    HAS_CAT = False

SEED = 42
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "SUM_FISI_10_21_clean.xlsx"

TARGET = "promedio_ponderado"

# === Candidatos que mencionaste ===
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

def norm_cols(df):
    df = df.copy()
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(" ", "_")
                  .str.replace("[^a-z0-9_]", "", regex=True))
    return df

def main():
    print(f"📥 Leyendo: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)
    df = norm_cols(df)

    # Filtrar por si alguna candidata no existe / está 100% NaN
    present = set(df.columns)
    num_present = [c for c in num_candidates if c in present and df[c].notna().any()]
    cat_present = [c for c in cat_candidates if c in present and df[c].notna().any()]

    needed = num_present + cat_present + [TARGET]
    df = df.dropna(subset=needed, how="any").reset_index(drop=True)

    # Cast numéricos
    for c in num_present + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    X = df[num_present + cat_present].copy()
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED
    )

    # Preprocesamiento (denso)
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer([
        ("num", cat_pipe if False else num_pipe, num_present),
        ("cat", cat_pipe, cat_present),
    ], remainder="drop", sparse_threshold=0.0)

    # Modelo base
    if HAS_CAT:
        model = CatBoostRegressor(
            iterations=800, depth=6, learning_rate=0.05, random_seed=SEED,
            loss_function="MAE", verbose=False
        )
        model_name = "CatBoostRegressor"
    else:
        model = RandomForestRegressor(
            n_estimators=600, random_state=SEED, n_jobs=-1
        )
        model_name = "RandomForestRegressor"

    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    print(f"✅ {model_name} MAE hold-out: {mae:.3f} (escala 0–20)")

    # === Permutation Importance a nivel de columna original ===
    print("\n🔎 Calculando Permutation Importance (por columna cruda)...")
    perm = permutation_importance(
        pipe, X_test, y_test, n_repeats=15, random_state=SEED, scoring="neg_mean_absolute_error"
    )
    # permutation_importance devuelve importancias por columna en el orden de X_test.columns
    fi = pd.DataFrame({
        "feature": X_test.columns,
        "importance": perm.importances_mean * -1.0,  # convertir a aumento de MAE
        "std": perm.importances_std
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Normalizar a porcentaje
    total = fi["importance"].clip(lower=0).sum()
    if total > 0:
        fi["pct_contrib"] = (fi["importance"].clip(lower=0) / total) * 100
    else:
        fi["pct_contrib"] = 0.0

    # Sugerencia: quedarnos con features hasta 95% de contribución acumulada
    fi["cum_pct"] = fi["pct_contrib"].cumsum()
    sugeridas = fi.loc[fi["cum_pct"] <= 95, "feature"].tolist()
    if not sugeridas:
        sugeridas = fi.head(3)["feature"].tolist()

    print("\n📊 Ranking (ΔMAE al permutar; mayor = más importante):")
    print(fi.to_string(index=False, formatters={
        "importance": "{:.4f}".format,
        "std": "{:.4f}".format,
        "pct_contrib": "{:.2f}%".format,
        "cum_pct": "{:.2f}%".format
    }))

    print("\n🧩 Propuesta de campos para el formulario (requeridos):")
    print(" - " + "\n - ".join(sugeridas))

    # Si el modelo es CatBoost, también mostramos importancias nativas
    if HAS_CAT:
        cat_model = pipe.named_steps["model"]
        # Importancias nativas están en el espacio transformado internamente,
        # pero CatBoost admite nombres originales si le pasas DataFrame con columnas.
        native_imp = pd.DataFrame({
            "feature": X.columns,
            "importance_native": cat_model.get_feature_importance(type="FeatureImportance")
        }).sort_values("importance_native", ascending=False).reset_index(drop=True)
        print("\n🌿 Importancias nativas CatBoost (mayor = más importante):")
        print(native_imp.to_string(index=False))

if __name__ == "__main__":
    main()
