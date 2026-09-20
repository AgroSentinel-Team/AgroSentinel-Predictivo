import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Carga de datos
df = pd.read_csv('data/smart_manufacturing_data.csv')

# ==========================================
# 2. PREPARACIÓN Y BALANCEO (SMOTE)
# ==========================================
columna_objetivo = 'anomaly_flag' 

columnas_a_ignorar = [
    columna_objetivo, 'timestamp', 'machine_id', 
    'downtime_risk', 'failure_type', 'maintenance_required', 'predicted_remaining_life'
]
columnas_a_borrar = [col for col in columnas_a_ignorar if col in df.columns]

X = df.drop(columnas_a_borrar, axis=1)
y = df[columna_objetivo]

print("Columnas exactas que usará el modelo para entrenar:", list(X.columns))

# --- EL PARCHE MÁGICO PARA TEXTO ---
columnas_texto = X.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in columnas_texto:
    X[col] = le.fit_transform(X[col].astype(str))

# División en Entrenamiento (80%) y Prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicamos SMOTE para balancear
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f"\nTotal de registros tras balanceo: {len(X_train_bal) + len(X_test)}")
print("Distribución de clases en entrenamiento:\n", y_train_bal.value_counts())


# ==========================================
# 3. ENTRENAMIENTO DE RANDOM FOREST
# ==========================================
print("\n--- Entrenando Random Forest Limpio ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_bal, y_train_bal)
rf_pred = rf_model.predict(X_test)

print(f"Precisión Random Forest: {accuracy_score(y_test, rf_pred) * 100:.2f}%\n")
print("Reporte Random Forest:")
print(classification_report(y_test, rf_pred))


# ==========================================
# 4. EXPORTACIÓN DEL MODELO A LA CARPETA DATA
# ==========================================
# Lo guardamos directamente en la carpeta data/ para que la API lo lea sin enredos
joblib.dump(rf_model, 'data/modelo_rf_final.pkl')
print("\n¡Modelo limpio exportado exitosamente en 'data/modelo_rf_final.pkl'!")