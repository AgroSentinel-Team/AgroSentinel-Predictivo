from datetime import datetime
from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# --- CONFIGURACIÓN DE LA BASE DE DATOS SQLITE ---
DATABASE_URL = "sqlite:///./data/agrosentinel_historial.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Definición de la Tabla Historial
class SensorLogDB(Base):
    __tablename__ = "historial_sensores"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    temperature = Column(Float)
    vibration = Column(Float)
    humidity = Column(Float)
    pressure = Column(Float)
    energy_consumption = Column(Float)
    machine_status = Column(Integer)
    prediccion_clase = Column(Integer)
    estado_motor = Column(String)
    confianza = Column(Float)
    componentes_afectados = Column(String)

# Creamos la base de datos y la tabla si no existen
Base.metadata.create_all(bind=engine)

# Dependencia para obtener la sesión de la base de datos en cada petición
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- CONFIGURACIÓN DE FASTAPI ---
app = FastAPI(
    title="AgroSentinel Predictive API",
    description="API para monitoreo predictivo, diagnóstico inteligente y persistencia de datos",
    version="3.0"
)

# Cargamos el modelo limpio
modelo = joblib.load('data/modelo_rf_final.pkl')

class DatosSensores(BaseModel):
    temperature: float
    vibration: float
    humidity: float
    pressure: float
    energy_consumption: float
    machine_status: int = Field(..., ge=0, le=1, description="Estado de la máquina: 0 para Apagado, 1 para Encendido")

def diagnosticar_fallas(datos: DatosSensores):
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
    return {"mensaje": "Motor monitoreado por AgroSentinel con base de datos de historial activa."}

@app.post("/predecir")
def predecir_falla(datos: DatosSensores, db: Session = Depends(get_db)):
    # 1. Preparamos los datos para el modelo de ML
    entrada_df = pd.DataFrame([{
        'temperature': datos.temperature,
        'vibration': datos.vibration,
        'humidity': datos.humidity,
        'pressure': datos.pressure,
        'energy_consumption': datos.energy_consumption,
        'machine_status': datos.machine_status
    }])
    
    # 2. Predicción de Machine Learning
    prediccion = int(modelo.predict(entrada_df)[0])
    probabilidad = float(modelo.predict_proba(entrada_df).max())
    
    # 3. Diagnóstico de componentes
    detalles_fallas = diagnosticar_fallas(datos)
    
    if prediccion == 1 or len(detalles_fallas) > 0:
        estado = "¡Alerta! Riesgo de Falla Crítica en el Motor"
        prediccion = 1
    else:
        estado = "Motor Operando con Normalidad"

    componentes_texto = ", ".join(detalles_fallas) if len(detalles_fallas) > 0 else "Ninguno (Parámetros estables)"

    # 4. GUARDAR EN LA BASE DE DATOS (HISTORIAL)
    nuevo_registro = SensorLogDB(
        temperature=datos.temperature,
        vibration=datos.vibration,
        humidity=datos.humidity,
        pressure=datos.pressure,
        energy_consumption=datos.energy_consumption,
        machine_status=datos.machine_status,
        prediccion_clase=prediccion,
        estado_motor=estado,
        confianza=probabilidad,
        componentes_afectados=componentes_texto
    )
    db.add(nuevo_registro)
    db.commit()
    db.refresh(nuevo_registro)

    # 5. Respuesta al cliente
    return {
        "id_registro_historial": nuevo_registro.id,
        "timestamp": nuevo_registro.timestamp,
        "prediccion_clase": prediccion,
        "estado_motor": estado,
        "confianza": probabilidad,
        "componentes_afectados": detalles_fallas if len(detalles_fallas) > 0 else ["Ninguno (Parámetros estables)"]
    }

# Endpoint extra para consultar todo el historial acumulado
@app.get("/historial")
def ver_historial(db: Session = Depends(get_db)):
    registros = db.query(SensorLogDB).all()
    return {
        "total_registros": len(registros),
        "datos": registros
    }