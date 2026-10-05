import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from sklearn.svm import SVC
from xgboost import XGBClassifier

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
# 3. ENTRENAMIENTO Y COMPARACIÓN DE MODELOS
# ==========================================
print("\n--- Entrenando y Comparando Modelos (Rúbrica MLOps) ---")

# A. Random Forest 
print("\n1. Entrenando Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
rf_model.fit(X_train_bal, y_train_bal)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Precisión Random Forest: {rf_acc * 100:.2f}%")

# B. Support Vector Machines (SVM)
print("\n2. Entrenando SVM (Support Vector Machine)...")
svm_model = SVC(kernel='rbf', random_state=42, class_weight='balanced')
svm_model.fit(X_train_bal, y_train_bal)
svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
print(f"Precisión SVM: {svm_acc * 100:.2f}%")

# C. XGBoost (Extreme Gradient Boosting)
print("\n3. Entrenando XGBoost...")
# XGBoost exige etiquetas numéricas. Usamos LabelEncoder solo para este modelo.
le_xgb = LabelEncoder()
y_train_xgb = le_xgb.fit_transform(y_train_bal)
y_test_xgb = le_xgb.transform(y_test)

xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train_bal, y_train_xgb)
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test_xgb, xgb_pred)
print(f"Precisión XGBoost: {xgb_acc * 100:.2f}%")

# ==========================================
# 4. EXPORTACIÓN DEL MODELO GANADOR
# ==========================================
joblib.dump(rf_model, 'data/modelo_rf_final.pkl')
print("\n¡Modelo multiclase GANADOR exportado exitosamente en 'data/modelo_rf_final.pkl'!")