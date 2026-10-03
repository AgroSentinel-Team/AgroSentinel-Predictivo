from datetime import datetime, timezone
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, func, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# --- CONFIGURACIÓN DE LA BASE DE DATOS SQLITE ---
DATABASE_URL = "sqlite:///./data/agrosentinel_historial.db"
BACKEND_RECORD_URL = os.getenv(
    "AGROSENTINEL_BACKEND_URL",
    "http://localhost:8080/api/predictive/records",
)
PREDICTIVE_API_TOKEN = os.getenv(
    "PREDICTIVE_API_TOKEN",
    "AgroSentinelPredictiveDevToken2026",
)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Definición de la Tabla Historial
class SensorLogDB(Base):
    __tablename__ = "historial_sensores"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    backend_record_id = Column(Integer, index=True, nullable=True) # ID de PostgreSQL
    machine_id = Column(String, index=True)
    asset_id = Column(Integer, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    temperature = Column(Float)
    vibration = Column(Float)
    humidity = Column(Float)
    pressure = Column(Float)
    energy_consumption = Column(Float)
    machine_status = Column(Integer)
    prediccion_clase = Column(Integer)
    tipo_falla_predicha = Column(String) # Etiqueta exacta de la IA
    estado_motor = Column(String)
    confianza = Column(Float)
    componentes_afectados = Column(String)
    falla_real_confirmada = Column(String, nullable=True) # Verdad absoluta para reentrenar

Base.metadata.create_all(bind=engine)

# Migración automática si faltan columnas
with engine.begin() as connection:
    columnas_actuales = {column["name"] for column in inspect(engine).get_columns("historial_sensores")}
    if "asset_id" not in columnas_actuales:
        connection.execute(text("ALTER TABLE historial_sensores ADD COLUMN asset_id INTEGER"))
    if "backend_record_id" not in columnas_actuales:
        connection.execute(text("ALTER TABLE historial_sensores ADD COLUMN backend_record_id INTEGER"))
    if "tipo_falla_predicha" not in columnas_actuales:
        connection.execute(text("ALTER TABLE historial_sensores ADD COLUMN tipo_falla_predicha VARCHAR"))
    if "falla_real_confirmada" not in columnas_actuales:
        connection.execute(text("ALTER TABLE historial_sensores ADD COLUMN falla_real_confirmada VARCHAR"))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- CONFIGURACIÓN DE FASTAPI ---
app = FastAPI(
    title="AgroSentinel Predictive API",
    description="API MLOps: Predicción Multiclase, Diagnóstico por Sensores y Feedback Loop",
    version="6.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo multiclase
modelo = joblib.load('data/modelo_rf_final.pkl')

# --- SCHEMAS (PYDANTIC) ---
class DatosSensores(BaseModel):
    asset_id: int = Field(..., gt=0, description="ID del activo registrado en AgroSentinel")
    machine_id: str = Field(..., description="ID predictivo asignado al activo, ej: MOTOR-12")
    temperature: float
    vibration: float
    humidity: float
    pressure: float
    energy_consumption: float
    machine_status: int = Field(..., ge=0, le=1)

class FeedbackRequest(BaseModel):
    fallaRealConfirmada: str = Field(..., description="Falla real observada por el técnico en campo")

# --- FUNCIONES DE DIAGNÓSTICO (HÍBRIDO) ---
def diagnosticar_fallas_sensores(datos: DatosSensores):
    anomalias = []
    
    # 1. TEMPERATURA
    if datos.temperature > 105.0:
        anomalias.append(f"T. Crítica ({datos.temperature}°C) - Revisar lubricación/ventilación")
    elif datos.temperature < 40.0:
        anomalias.append(f"T. Baja ({datos.temperature}°C) - Operación en vacío/Fallo de sensor")
        
    # 2. VIBRACIÓN
    if datos.vibration > 90.0:
        anomalias.append(f"Vibración Alta ({datos.vibration} Hz) - Revisar desalineación/rodamientos")
        
    # 3. HUMEDAD
    if datos.humidity > 75.0:
        anomalias.append(f"Humedad Alta ({datos.humidity}%) - Riesgo de condensación")
        
    # 4. PRESIÓN
    if datos.pressure > 4.5:
        anomalias.append(f"Presión Elevada ({datos.pressure} bar) - Válvulas obstruidas")
    elif datos.pressure < 1.5:
        anomalias.append(f"Presión Baja ({datos.pressure} bar) - Fugas/Filtros tapados")
        
    # 5. CONSUMO DE ENERGÍA
    if datos.energy_consumption > 4.5:
        anomalias.append(f"Consumo Alto ({datos.energy_consumption} kW) - Sobrecarga mecánica")
        
    return anomalias

def obtener_explicacion_falla_ia(falla_ia: str):
    causas = {
        "overheating": "General: Fricción o sobrecarga.",
        "vibration issue": "General: Desbalanceo o anclajes sueltos.",
        "pressure drop": "General: Fallo en bomba de suministro.",
        "electrical fault": "General: Aislamiento o cortocircuitos."
    }
    return causas.get(falla_ia.strip().lower(), "Revisión requerida.")

def guardar_en_backend(registro: dict):
    request = Request(
        BACKEND_RECORD_URL,
        data=json.dumps(registro).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Predictive-Token": PREDICTIVE_API_TOKEN,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            try:
                response_body = json.loads(response.read().decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise HTTPException(
                    status_code=502,
                    detail="Spring Boot devolvió una respuesta JSON inválida al guardar la predicción.",
                ) from error

            backend_record_id = response_body.get("id") if isinstance(response_body, dict) else None
            if type(backend_record_id) is not int or backend_record_id < 1:
                raise HTTPException(
                    status_code=502,
                    detail="Spring Boot no devolvió el ID de la predicción guardada.",
                )
            return backend_record_id
    except HTTPError as error:
        try:
            response_body = json.loads(error.read().decode("utf-8"))
            upstream_detail = (
                response_body.get("detail") or response_body.get("message") or response_body.get("error")
                if isinstance(response_body, dict)
                else None
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            upstream_detail = None
        detail = upstream_detail or f"Spring Boot rechazó la predicción (HTTP {error.code})."
        raise HTTPException(
            status_code=502,
            detail=f"No fue posible guardar la predicción en AgroSentinel: {detail}",
        ) from error
    except (URLError, TimeoutError) as error:
        raise HTTPException(
            status_code=502,
            detail="No fue posible conectar con el backend de AgroSentinel.",
        ) from error

# --- ENDPOINTS ---
@app.get("/")
def home():
    return {"mensaje": "AgroSentinel Predictive API en línea. Ciclo MLOps y Diagnóstico Híbrido activos."}

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
    
    # 1. PREDICCIÓN DE IA
    tipo_falla_ia = str(modelo.predict(entrada_df)[0])
    probabilidad = float(modelo.predict_proba(entrada_df).max())
    
    # 2. DIAGNÓSTICO FÍSICO DE LÍMITES
    alertas_sensores = diagnosticar_fallas_sensores(datos)
    
    # 3. EVALUACIÓN COMBINADA
    estado_ia_seguro = tipo_falla_ia.strip().lower() in ["normal", "no failure", "none"]
    
    # Hay falla si la IA lo dice OR si algún sensor pasó el límite físico
    if estado_ia_seguro and len(alertas_sensores) == 0:
        estado = "Motor Operando con Normalidad"
        prediccion_num = 0  
        componentes_texto = "Ninguno (Parámetros estables)"
    else:
        estado = "¡Alerta! Riesgo Detectado en Motor"
        prediccion_num = 1  
        
        mensajes_finales = []
        # Si la IA detecta algo anormal, agregamos su diagnóstico
        if not estado_ia_seguro:
            explicacion_ia = obtener_explicacion_falla_ia(tipo_falla_ia)
            mensajes_finales.append(f"IA: {tipo_falla_ia} ({probabilidad*100:.1f}%) -> {explicacion_ia}")
        
        # Si los sensores superan límites, los listamos explícitamente
        if len(alertas_sensores) > 0:
            mensajes_finales.append("Sensores Críticos: " + " | ".join(alertas_sensores))
            
        componentes_texto = " || ".join(mensajes_finales)

    timestamp = datetime.now(timezone.utc).replace(tzinfo=None)

    # 4. GUARDAR EN SPRING BOOT (Y recuperar el ID)
    pg_id = guardar_en_backend({
        "assetId": datos.asset_id,
        "machineId": datos.machine_id,
        "timestamp": timestamp.isoformat(),
        "temperature": datos.temperature,
        "vibration": datos.vibration,
        "humidity": datos.humidity,
        "pressure": datos.pressure,
        "energyConsumption": datos.energy_consumption,
        "machineStatus": datos.machine_status,
        "predictionClass": prediccion_num,
        "tipoFallaPredicha": tipo_falla_ia,
        "motorState": estado,
        "confidence": probabilidad,
        "affectedComponents": componentes_texto,
    })

    # 5. GUARDAR EN SQLITE LOCAL (Conectando el ID de Spring)
    nuevo_registro = SensorLogDB(
        backend_record_id=pg_id,
        asset_id=datos.asset_id,
        machine_id=datos.machine_id,
        timestamp=timestamp,
        temperature=datos.temperature,
        vibration=datos.vibration,
        humidity=datos.humidity,
        pressure=datos.pressure,
        energy_consumption=datos.energy_consumption,
        machine_status=datos.machine_status,
        prediccion_clase=prediccion_num,
        tipo_falla_predicha=tipo_falla_ia,
        estado_motor=estado,
        confianza=probabilidad,
        componentes_afectados=componentes_texto
    )
    db.add(nuevo_registro)
    db.commit()
    db.refresh(nuevo_registro)

    return {
        "id_registro_historial": nuevo_registro.id,
        "backend_record_id": pg_id,
        "asset_id": datos.asset_id,
        "machine_id": nuevo_registro.machine_id,
        "timestamp": nuevo_registro.timestamp,
        "prediccion_clase": prediccion_num,
        "tipo_falla_predicha": tipo_falla_ia,
        "estado_motor": estado,
        "confianza": probabilidad,
        "componentes_afectados": [componentes_texto]
    }

@app.put("/historial/{id}/feedback")
def guardar_feedback(id: int, feedback: FeedbackRequest, db: Session = Depends(get_db)):
    registro = db.query(SensorLogDB).filter(SensorLogDB.backend_record_id == id).first()
    
    if not registro:
        raise HTTPException(status_code=404, detail=f"No se encontró el registro con backend_id {id} en SQLite.")
        
    registro.falla_real_confirmada = feedback.fallaRealConfirmada
    db.commit()
    
    return {
        "mensaje": "Feedback guardado exitosamente en SQLite", 
        "falla_real": registro.falla_real_confirmada
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