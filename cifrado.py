import torch

try:
    datos = torch.load('data/modelo_lstm_final.pt', map_location='cpu', weights_only=False)
except TypeError:
    datos = torch.load('data/modelo_lstm_final.pt', map_location='cpu')

print({
    "classes": datos["classes"],
    "sequence_length": datos["sequence_length"],
    "best_epoch": datos["best_epoch"],
    "validation_accuracy": datos["validation_accuracy"],
})