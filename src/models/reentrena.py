import sqlite3

import pandas as pd

from src.models.entrenar_lstm import MODEL_PATH, apply_physical_failure_labels, train_and_save


def main():
    print("Iniciando reentrenamiento de la LSTM...")
    print("Cargando datos originales...")
    df_original = pd.read_csv("data/smart_manufacturing_data.csv")
    df_original = apply_physical_failure_labels(df_original)
    print("Extrayendo lecturas confirmadas por técnicos desde SQLite...")
    with sqlite3.connect("./data/agrosentinel_historial.db") as connection:
        df_feedback = pd.read_sql_query(
            """
            SELECT
                machine_id,
                timestamp,
                temperature,
                vibration,
                humidity,
                pressure,
                energy_consumption,
                machine_status,
                falla_real_confirmada AS failure_type
            FROM historial_sensores
            WHERE falla_real_confirmada IS NOT NULL
                AND TRIM(falla_real_confirmada) <> ''
            """,
            connection,
        )

    if df_feedback.empty:
        raise ValueError("No hay feedback técnico confirmado; el modelo no se reentrenó.")

    print(f"Feedback técnico encontrado: {len(df_feedback)} registros.")
    df_combined = pd.concat([df_original, df_feedback], ignore_index=True)

    print("Entrenando LSTM con datos originales y feedback técnico...")
    train_and_save(df_combined, MODEL_PATH)
    print("Reentrenamiento LSTM completado.")


if __name__ == "__main__":
    main()
