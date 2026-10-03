import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Iniciando proceso de REENTRENAMIENTO...")

# 1. Cargar el dataset original (tu base de conocimiento inicial)
print("1. Cargando datos originales del CSV...")
df_original = pd.read_csv('data/smart_manufacturing_data.csv')

# Filtramos las columnas que no nos sirven del CSV original, tal como hicimos antes
columnas_a_borrar = ['anomaly_flag', 'timestamp', 'machine_id', 'downtime_risk', 'maintenance_required', 'predicted_remaining_life']
df_original = df_original.drop(columns=[col for col in columnas_a_borrar if col in df_original.columns])

# 2. Cargar los datos NUEVOS validados por los técnicos desde SQLite
print("2. Extrayendo feedback real de la base de datos...")
# Nos conectamos a la base de datos de tu API
conexion = sqlite3.connect('./data/agrosentinel_historial.db')

# Traemos SOLAMENTE los registros donde un humano confirmó la falla
query = """
    SELECT 
        temperature, 
        vibration, 
        humidity, 
        pressure, 
        energy_consumption, 
        machine_status, 
        falla_real_confirmada AS failure_type 
    FROM historial_sensores 
    WHERE falla_real_confirmada IS NOT NULL
"""
df_nuevos_datos = pd.read_sql_query(query, conexion)
conexion.close()

if len(df_nuevos_datos) == 0:
    print("❌ Aún no hay datos de feedback confirmados en la base de datos. No se puede reentrenar.")
    exit()

print(f"✅ Se encontraron {len(df_nuevos_datos)} registros confirmados por técnicos.")

# 3. Unir la experiencia antigua con la nueva
print("3. Fusionando conocimientos...")
df_combinado = pd.concat([df_original, df_nuevos_datos], ignore_index=True)

# 4. Preparar X e y
X = df_combinado.drop('failure_type', axis=1)
y = df_combinado['failure_type']

# Convertimos texto a números en X si es necesario
columnas_texto = X.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in columnas_texto:
    X[col] = le.fit_transform(X[col].astype(str))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balanceo
print("4. Balanceando el nuevo dataset...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# 5. Reentrenar el modelo
print("5. Entrenando el modelo actualizado...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_bal, y_train_bal)
rf_pred = rf_model.predict(X_test)

print(f"\nPrecisión del NUEVO modelo: {accuracy_score(y_test, rf_pred) * 100:.2f}%\n")
print("Reporte de clasificación actualizado:")
print(classification_report(y_test, rf_pred))

# 6. Sobrescribir el modelo antiguo
joblib.dump(rf_model, 'data/modelo_rf_final.pkl')
print("\n ¡Reentrenamiento exitoso! El archivo 'modelo_rf_final.pkl' ha sido actualizado.")