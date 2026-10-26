import copy
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.lstm_model import LSTMClassifier

FEATURE_COLUMNS = [
    "temperature",
    "vibration",
    "humidity",
    "pressure",
    "energy_consumption",
    "machine_status",
]
TARGET_COLUMN = "failure_type"
SEQUENCE_LENGTH = 10
HIDDEN_SIZE = 64
NUM_LAYERS = 2
BATCH_SIZE = 512
MAX_EPOCHS = 25
PATIENCE = 5
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = str(DATA_DIR / "modelo_lstm_final.pt")


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


def make_temporal_split(df):
    required_columns = ["machine_id", "timestamp", *FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas para entrenar la LSTM: {missing_columns}")

    clean_df = df[required_columns].dropna().copy()
    clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], errors="coerce")
    clean_df = clean_df.dropna(subset=["timestamp"])
    clean_df = clean_df.sort_values(["machine_id", "timestamp"])
    if clean_df.empty:
        raise ValueError("No hay lecturas completas con fecha, máquina, sensores y etiqueta.")

    classes = sorted(clean_df[TARGET_COLUMN].astype(str).unique().tolist())
    class_to_index = {label: index for index, label in enumerate(classes)}
    train_sequences = []
    train_targets = []
    valid_sequences = []
    valid_targets = []
    train_feature_rows = []

    for _, machine_df in clean_df.groupby("machine_id", sort=False):
        machine_df = machine_df.drop_duplicates("timestamp", keep="last")
        if len(machine_df) <= SEQUENCE_LENGTH:
            continue

        values = machine_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        labels = machine_df[TARGET_COLUMN].astype(str).map(class_to_index).to_numpy(dtype=np.int64)
        split_index = max(SEQUENCE_LENGTH, int(len(machine_df) * 0.8))
        train_feature_rows.append(values[:split_index])

        for target_index in range(SEQUENCE_LENGTH - 1, len(machine_df)):
            sequence = values[target_index - SEQUENCE_LENGTH + 1 : target_index + 1]
            if target_index < split_index:
                train_sequences.append(sequence)
                train_targets.append(labels[target_index])
            else:
                valid_sequences.append(sequence)
                valid_targets.append(labels[target_index])

    if not train_sequences or not valid_sequences:
        raise ValueError("No se pudieron crear ventanas temporales de entrenamiento y validación.")

    scaler_mean = np.concatenate(train_feature_rows).mean(axis=0)
    scaler_std = np.concatenate(train_feature_rows).std(axis=0)
    scaler_std[scaler_std == 0] = 1.0

    return {
        "X_train": ((np.asarray(train_sequences) - scaler_mean) / scaler_std).astype(np.float32),
        "y_train": np.asarray(train_targets, dtype=np.int64),
        "X_valid": ((np.asarray(valid_sequences) - scaler_mean) / scaler_std).astype(np.float32),
        "y_valid": np.asarray(valid_targets, dtype=np.int64),
        "classes": classes,
        "scaler_mean": scaler_mean.astype(np.float32),
        "scaler_std": scaler_std.astype(np.float32),
    }


def train_model(X_train, y_train, X_valid, y_valid, num_classes):
    torch.manual_seed(42)
    model = LSTMClassifier(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
    )

    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    class_weights /= class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

    training_data = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    training_loader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_x = torch.from_numpy(X_valid)
    validation_y = y_valid

    best_state = None
    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for sequences, targets in training_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(sequences), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(targets)

        model.eval()
        validation_predictions = []
        with torch.inference_mode():
            for start in range(0, len(validation_x), BATCH_SIZE):
                logits = model(validation_x[start : start + BATCH_SIZE])
                validation_predictions.extend(logits.argmax(dim=1).tolist())

        macro_f1 = f1_score(
            validation_y,
            validation_predictions,
            labels=np.arange(num_classes),
            average="macro",
            zero_division=0,
        )
        validation_accuracy = accuracy_score(validation_y, validation_predictions)
        average_loss = total_loss / len(training_data)
        print(
            f"Época {epoch:02d}: pérdida={average_loss:.4f}, "
            f"exactitud_validación={validation_accuracy * 100:.2f}%, "
            f"F1_macro={macro_f1:.4f}"
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("El entrenamiento LSTM no produjo un estado de modelo válido.")

    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def train_and_save(df, model_path=MODEL_PATH):
    split = make_temporal_split(df)
    print(
        f"Ventanas LSTM: {len(split['y_train'])} entrenamiento, "
        f"{len(split['y_valid'])} validación; secuencia={SEQUENCE_LENGTH}."
    )
    model, best_epoch = train_model(
        split["X_train"],
        split["y_train"],
        split["X_valid"],
        split["y_valid"],
        len(split["classes"]),
    )

    model.eval()
    predictions = []
    with torch.inference_mode():
        X_valid = torch.from_numpy(split["X_valid"])
        for start in range(0, len(X_valid), BATCH_SIZE):
            predictions.extend(model(X_valid[start : start + BATCH_SIZE]).argmax(dim=1).tolist())

    accuracy = accuracy_score(split["y_valid"], predictions)
    print(f"\nExactitud de validación temporal LSTM: {accuracy * 100:.2f}%")
    print(
        classification_report(
            split["y_valid"],
            predictions,
            labels=np.arange(len(split["classes"])),
            target_names=split["classes"],
            zero_division=0,
        )
    )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "classes": split["classes"],
            "scaler_mean": split["scaler_mean"].tolist(),
            "scaler_std": split["scaler_std"].tolist(),
            "sequence_length": SEQUENCE_LENGTH,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "num_classes": len(split["classes"]),
            "best_epoch": best_epoch,
            "validation_accuracy": float(accuracy),
        },
        model_path,
    )
    print(f"Modelo LSTM guardado en '{model_path}'.")
    return model_path


def main():
    print("Cargando lecturas históricas...")
    dataset_path = DATA_DIR / "smart_manufacturing_data.csv"
    df = pd.read_csv(dataset_path)
    print(f"Registros: {len(df)}; máquinas: {df['machine_id'].nunique()}")
    print("Generando etiquetas objetivo a partir de las reglas físicas del servicio...")
    df = apply_physical_failure_labels(df)
    train_and_save(df, MODEL_PATH)


if __name__ == "__main__":
    main()
