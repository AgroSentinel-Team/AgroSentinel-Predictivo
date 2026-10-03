import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

os.makedirs('data', exist_ok=True)

# 1. Carga de datos
print("Cargando datos...")
df = pd.read_csv('data/smart_manufacturing_data.csv')
print(f"Total de registros antes de limpieza: {len(df)}")

# ==========================================
# 1.5. SANITIZACIÓN BASADA EN REGLAS FÍSICAS
# ==========================================
print("\nAplicando reglas físicas para corregir las etiquetas de falla...")

# Si varias reglas se activan en una fila, prevalece la primera regla de esta lista.
condiciones = [
    df['temperature'] > 105.0,
    df['vibration'] > 90.0,
    (df['pressure'] > 4.5) | (df['pressure'] < 1.5),
    df['energy_consumption'] > 4.5,
]
etiquetas = [
    'Overheating',
    'Vibration Issue',
    'Pressure Drop',
    'Electrical Fault',
]

df['failure_type'] = np.select(condiciones, etiquetas, default='Normal')
print("Distribución de etiquetas sanitizadas:")
print(df['failure_type'].value_counts().to_string())

# ==========================================
# 2. PREPARACIÓN Y BALANCEO (SMOTE)
# ==========================================
columna_objetivo = 'failure_type' 

columnas_a_ignorar = [
    columna_objetivo,
    'anomaly_flag', 
    'timestamp', 
    'machine_id', 
    'downtime_risk', 
    'maintenance_required', 
    'predicted_remaining_life'
]
columnas_a_borrar = [col for col in columnas_a_ignorar if col in df.columns]

X = df.drop(columnas_a_borrar, axis=1)
y = df[columna_objetivo]

print("\nColumnas que usará el modelo:", list(X.columns))

# Convertir texto a números en X
columnas_texto = X.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in columnas_texto:
    X[col] = le.fit_transform(X[col].astype(str))

# División en Entrenamiento y Prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nAplicando balanceo SMOTE...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# ==========================================
# 3. ENTRENAMIENTO DE RANDOM FOREST
# ==========================================
print("\n--- Entrenando Random Forest Multiclase Limpio ---")
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced',
)
rf_model.fit(X_train_bal, y_train_bal)
rf_pred = rf_model.predict(X_test)

print(f"\nPrecisión Random Forest: {accuracy_score(y_test, rf_pred) * 100:.2f}%\n")
print("Reporte Random Forest:")
print(classification_report(y_test, rf_pred))

# ==========================================
# 4. EXPORTACIÓN DEL MODELO
# ==========================================
joblib.dump(rf_model, 'data/modelo_rf_final.pkl')
print("\n¡Modelo multiclase LIMPIO exportado exitosamente en 'data/modelo_rf_final.pkl'!")