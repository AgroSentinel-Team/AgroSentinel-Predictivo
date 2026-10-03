import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Asegurarnos de que exista la carpeta data/
os.makedirs('data', exist_ok=True)

# 1. Carga de datos
print("Cargando datos...")
df = pd.read_csv('data/smart_manufacturing_data.csv')

# ==========================================
# 2. PREPARACIÓN Y BALANCEO (SMOTE)
# ==========================================
# CAMBIO CLAVE: Ahora nuestro objetivo es el TIPO de falla
columna_objetivo = 'failure_type' 

columnas_a_ignorar = [
    columna_objetivo, # ¡CRÍTICO! Hay que borrar la respuesta de los datos de entrenamiento (X)
    'anomaly_flag',   # Borramos el flag binario porque ya no lo necesitamos
    'timestamp', 
    'machine_id', 
    'downtime_risk', 
    'maintenance_required', 
    'predicted_remaining_life'
]
columnas_a_borrar = [col for col in columnas_a_ignorar if col in df.columns]

X = df.drop(columnas_a_borrar, axis=1)
y = df[columna_objetivo]

print("Columnas exactas que usará el modelo para entrenar:", list(X.columns))

# --- EL PARCHE MÁGICO PARA TEXTO EN X ---
# Transformamos cualquier columna de texto a números (si quedó alguna)
columnas_texto = X.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in columnas_texto:
    X[col] = le.fit_transform(X[col].astype(str))

# NOTA: NO le aplicamos LabelEncoder a 'y' (failure_type) para que el modelo 
# nos devuelva el texto exacto (ej. "Overheating") y no un número sin sentido.

# División en Entrenamiento (80%) y Prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicamos SMOTE para balancear
print("Aplicando balanceo SMOTE...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f"\nTotal de registros tras balanceo: {len(X_train_bal) + len(X_test)}")
print("Distribución de clases en entrenamiento:\n", y_train_bal.value_counts())

# ==========================================
# 3. ENTRENAMIENTO DE RANDOM FOREST
# ==========================================
print("\n--- Entrenando Random Forest Multiclase ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_bal, y_train_bal)
rf_pred = rf_model.predict(X_test)

print(f"\nPrecisión Random Forest: {accuracy_score(y_test, rf_pred) * 100:.2f}%\n")
print("Reporte Random Forest:")
print(classification_report(y_test, rf_pred))

# ==========================================
# 4. EXPORTACIÓN DEL MODELO
# ==========================================
joblib.dump(rf_model, 'data/modelo_rf_final.pkl')
print("\n¡Modelo multiclase exportado exitosamente en 'data/modelo_rf_final.pkl'!")