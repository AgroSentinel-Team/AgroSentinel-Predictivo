from pathlib import Path

import numpy as np
import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, sequences):
        _, (hidden, _) = self.lstm(sequences)
        return self.classifier(hidden[-1])


def load_lstm_model(model_path):
    resolved_path = str(Path(model_path).expanduser().resolve())
    try:
        artifact = torch.load(resolved_path, map_location="cpu", weights_only=False)
    except TypeError:
        artifact = torch.load(resolved_path, map_location="cpu")

    model = LSTMClassifier(
        input_size=len(artifact["feature_columns"]),
        hidden_size=artifact["hidden_size"],
        num_layers=artifact["num_layers"],
        num_classes=artifact["num_classes"],
    )
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model, artifact


def predict_lstm(model, artifact, readings):
    if not readings:
        raise ValueError("Se requiere al menos una lectura para predecir.")

    feature_columns = artifact["feature_columns"]
    sequence_length = artifact["sequence_length"]
    first_reading = readings[0]
    padded_readings = [first_reading] * max(0, sequence_length - len(readings)) + readings
    padded_readings = padded_readings[-sequence_length:]

    values = np.asarray(
        [[reading[column] for column in feature_columns] for reading in padded_readings],
        dtype=np.float32,
    )
    mean = np.asarray(artifact["scaler_mean"], dtype=np.float32)
    std = np.asarray(artifact["scaler_std"], dtype=np.float32)
    tensor = torch.from_numpy(((values - mean) / std)[np.newaxis, ...])

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    predicted_index = int(probabilities.argmax().item())
    return artifact["classes"][predicted_index], float(probabilities[predicted_index].item())
