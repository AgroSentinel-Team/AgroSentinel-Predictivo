from datetime import datetime
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, func
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
    machine_id = Column(String, index=True)
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

Base.metadata.create_all(bind=engine)

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
    version="3.3"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

modelo = joblib.load('data/modelo_rf_final.pkl')

class DatosSensores(BaseModel):
    machine_id: str = Field(..., description="ID único del equipo, ej: EQ-001")
    temperature: float
    vibration: float
    humidity: float
    pressure: float
    energy_consumption: float
    machine_status: int = Field(..., ge=0, le=1)

def diagnosticar_fallas(datos: DatosSensores):
    anomalias = []
    
    if datos.temperature > 105.0:
        anomalias.append(f"Temperatura crítica ({datos.temperature}°C)")
    elif datos.temperature < 40.0:
        anomalias.append(f"Temperatura muy baja ({datos.temperature}°C)")
        
    if datos.vibration > 90.0:
        anomalias.append(f"Vibración excesiva ({datos.vibration} Hz)")
        
    if datos.humidity > 75.0:
        anomalias.append(f"Humedad muy alta ({datos.humidity}%)")
        
    if datos.pressure > 4.5:
        anomalias.append(f"Presión elevada ({datos.pressure} bar)")
    elif datos.pressure < 1.5:
        anomalias.append(f"Presión baja ({datos.pressure} bar)")
        
    if datos.energy_consumption > 4.5:
        anomalias.append(f"Consumo de energía alto ({datos.energy_consumption} kW)")
        
    if datos.machine_status == 0 and (datos.temperature > 90 or datos.vibration > 80):
        anomalias.append("Motor apagado reportando estrés térmico/mecánico")

    return anomalias

@app.get("/")
def home():
    return {"mensaje": "Motor monitoreado por AgroSentinel con base de datos de historial activa."}

@app.post("/predecir")
def predecir_falla(datos: DatosSensores, db: Session = Depends(get_db)):
    entrada_df = pd.DataFrame([{
        'temperature': datos.temperature,
        'vibration': datos.vibration,
        'humidity': datos.humidity,
        'pressure': datos.pressure,
        'energy_consumption': datos.energy_consumption,
        'machine_status': datos.machine_status
    }])
    
    prediccion = int(modelo.predict(entrada_df)[0])
    probabilidad = float(modelo.predict_proba(entrada_df).max())
    detalles_fallas = diagnosticar_fallas(datos)
    
    # NUEVA LÓGICA: Inyección de parámetros reales en la alerta de IA
    if prediccion == 1 and len(detalles_fallas) == 0:
        resumen_parametros = f"T: {datos.temperature}°C | Vib: {datos.vibration}Hz | Hum: {datos.humidity}% | Pres: {datos.pressure}bar"
        detalles_fallas.append(f"Alerta IA ({probabilidad*100:.1f}%): Combinación riesgosa detectada en -> {resumen_parametros}")
    
    if prediccion == 1 or len(detalles_fallas) > 0:
        estado = "¡Alerta! Riesgo de Falla Crítica en el Motor"
        prediccion = 1
    else:
        estado = "Motor Operando con Normalidad"

    componentes_texto = " + ".join(detalles_fallas) if len(detalles_fallas) > 0 else "Ninguno (Parámetros estables)"

    nuevo_registro = SensorLogDB(
        machine_id=datos.machine_id,
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

    return {
        "id_registro_historial": nuevo_registro.id,
        "machine_id": nuevo_registro.machine_id,
        "timestamp": nuevo_registro.timestamp,
        "prediccion_clase": prediccion,
        "estado_motor": estado,
        "confianza": probabilidad,
        "componentes_afectados": detalles_fallas if len(detalles_fallas) > 0 else ["Ninguno (Parámetros estables)"]
    }

@app.get("/historial")
def ver_historial(db: Session = Depends(get_db)):
    registros = db.query(SensorLogDB).all()
    return {
        "total_registros": len(registros),
        "datos": registros
    }

@app.get("/equipos/estado")
def obtener_estado_equipos(db: Session = Depends(get_db)):
    subquery = db.query(
        SensorLogDB.machine_id, 
        func.max(SensorLogDB.id).label("max_id")
    ).group_by(SensorLogDB.machine_id).subquery()
    
    ultimos_registros = db.query(SensorLogDB).join(
        subquery, SensorLogDB.id == subquery.c.max_id
    ).all()
    
    return {"equipos": ultimos_registros}