from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

FEATURE_COLUMNS = [
    "temperature",
    "vibration",
    "humidity",
    "pressure",
    "energy_consumption",
    "machine_status",
]
TARGET_COLUMN = "failure_type"
RANDOM_STATE = 42
SVM_MAX_SAMPLES_PER_CLASS = 2_000
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def apply_physical_failure_labels(df):
    result = df.copy()
    conditions = [
        result["temperature"] > 105.0,
        result["vibration"] > 90.0,
        (result["pressure"] > 4.5) | (result["pressure"] < 1.5),
        result["energy_consumption"] > 4.5,
    ]
    labels = [
        "Overheating",
        "Vibration Issue",
        "Pressure Drop",
        "Electrical Fault",
    ]
    result[TARGET_COLUMN] = np.select(conditions, labels, default="Normal")
    return result


def train_and_save(df, data_dir=DATA_DIR):
    required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas para entrenar modelos tabulares: {missing_columns}")

    clean_df = df[required_columns].replace([np.inf, -np.inf], np.nan).dropna()
    if clean_df.empty:
        raise ValueError("No hay registros completos para entrenar los modelos tabulares.")

    X = clean_df[FEATURE_COLUMNS]
    y = clean_df[TARGET_COLUMN].astype(str)
    class_counts = y.value_counts()
    if class_counts.min() < 2:
        raise ValueError(
            "SMOTE requiere al menos dos registros en cada clase; "
            f"conteos encontrados: {class_counts.to_dict()}"
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    smote = SMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=min(5, int(class_counts.min()) - 1),
    )
    X_resampled_scaled, y_resampled = smote.fit_resample(X_scaled, y)
    X_resampled = pd.DataFrame(
        scaler.inverse_transform(X_resampled_scaled),
        columns=FEATURE_COLUMNS,
    )

    random_forest = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    random_forest.fit(X_resampled, y_resampled)

    svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", random_state=RANDOM_STATE)),
        ]
    )
    balanced_labels = np.asarray(y_resampled)
    rng = np.random.default_rng(RANDOM_STATE)
    svm_indices = np.concatenate(
        [
            rng.choice(
                np.flatnonzero(balanced_labels == label),
                size=min(SVM_MAX_SAMPLES_PER_CLASS, np.count_nonzero(balanced_labels == label)),
                replace=False,
            )
            for label in np.unique(balanced_labels)
        ]
    )
    svm.fit(X_resampled.iloc[svm_indices], balanced_labels[svm_indices])

    label_encoder = LabelEncoder()
    y_xgb = label_encoder.fit_transform(y_resampled)
    xgboost = XGBClassifier(
        n_estimators=200,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgboost.fit(X_resampled, y_xgb)

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    model_paths = {
        "rf": data_dir / "modelo_rf.pkl",
        "svm": data_dir / "modelo_svm.pkl",
        "xgb": data_dir / "modelo_xgb.pkl",
    }
    joblib.dump(random_forest, model_paths["rf"])
    joblib.dump(svm, model_paths["svm"])
    joblib.dump(
        {
            "model": xgboost,
            "label_encoder": label_encoder,
            "feature_columns": FEATURE_COLUMNS,
        },
        model_paths["xgb"],
    )

    for model_name, model_path in model_paths.items():
        print(f"Modelo {model_name.upper()} guardado en '{model_path}'.")
    return model_paths


def main():
    dataset_path = DATA_DIR / "smart_manufacturing_data.csv"
    print(f"Cargando datos desde '{dataset_path}'...")
    df = pd.read_csv(dataset_path)
    print("Generando etiquetas objetivo a partir de las reglas físicas...")
    df = apply_physical_failure_labels(df)
    train_and_save(df)


if __name__ == "__main__":
    main()
