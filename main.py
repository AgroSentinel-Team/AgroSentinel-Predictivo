from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(
    title="AgroSentinel Predictive API",
    description="API para monitoreo predictivo y diagnóstico avanzado del motor agrícola",
    version="2.1"
)

# Cargamos el modelo limpio
modelo = joblib.load('data/modelo_rf_final.pkl')

class DatosSensores(BaseModel):
    temperature: float
    vibration: float
    humidity: float
    pressure: float
    energy_consumption: float
    # Restringimos estrictamente para que solo acepte 0 (Apagado) o 1 (Encendido)
    machine_status: int = Field(..., ge=0, le=1, description="Estado de la máquina: 0 para Apagado, 1 para Encendido")

def diagnosticar_fallas(datos: DatosSensores):
    """Función de reglas de negocio para identificar qué sensores están causando el problema"""
    anomalias = []
    
    if datos.temperature > 105.0:
        anomalias.append(f"Temperatura crítica ({datos.temperature}°C - Riesgo de sobrecalentamiento)")
    elif datos.temperature < 40.0:
        anomalias.append(f"Temperatura muy baja ({datos.temperature}°C - Posible fallo en arranque)")
        
    if datos.vibration > 90.0:
        anomalias.append(f"Vibración excesiva ({datos.vibration} Hz - Posible daño mecánico o desalineación)")
        
    if datos.humidity > 75.0:
        anomalias.append(f"Humedad ambiental muy alta ({datos.humidity}% - Riesgo de cortocircuito o condensación)")
        
    if datos.pressure > 4.5:
        anomalias.append(f"Presión elevada ({datos.pressure} bar - Sobrecarga en el sistema)")
    elif datos.pressure < 1.5:
        anomalias.append(f"Presión baja ({datos.pressure} bar - Posible fuga o fallo de bomba)")
        
    if datos.energy_consumption > 4.5:
        anomalias.append(f"Consumo de energía alto ({datos.energy_consumption} kW - Esfuerzo excesivo del motor)")
        
    if datos.machine_status == 0 and (datos.temperature > 90 or datos.vibration > 80):
        anomalias.append("Incoherencia operacional: El motor reporta estado apagado/inactivo pero registra valores de estrés elevados")

    return anomalias

@app.get("/")
def home():
    return {"mensaje": "Motor monitoreado por AgroSentinel con diagnóstico inteligente en línea."}

@app.post("/predecir")
def predecir_falla(datos: DatosSensores):
    # Creamos el DataFrame para el modelo de ML
    entrada_df = pd.DataFrame([{
        'temperature': datos.temperature,
        'vibration': datos.vibration,
        'humidity': datos.humidity,
        'pressure': datos.pressure,
        'energy_consumption': datos.energy_consumption,
        'machine_status': datos.machine_status
    }])
    
    # Predicción de Machine Learning
    prediccion = int(modelo.predict(entrada_df)[0])
    probabilidad = float(modelo.predict_proba(entrada_df).max())
    
    # Ejecutamos el diagnóstico de componentes específicos
    detalles_fallas = diagnosticar_fallas(datos)
    
    # Determinamos el estado final combinando ML y lógica de negocio
    if prediccion == 1 or len(detalles_fallas) > 0:
        estado = "¡Alerta! Riesgo de Falla Crítica en el Motor"
        prediccion = 1
    else:
        estado = "Motor Operando con Normalidad"

    return {
        "prediccion_clase": prediccion,
        "estado_motor": estado,
        "confianza": probabilidad,
        "componentes_afectados": detalles_fallas if len(detalles_fallas) > 0 else ["Ninguno (Parámetros estables)"]
    }