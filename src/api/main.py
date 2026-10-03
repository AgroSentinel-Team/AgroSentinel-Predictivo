from datetime import datetime, timezone
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, Depends, Header, HTTPException
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
    machine_id = Column(String, index=True)
    asset_id = Column(Integer, index=True)
    backend_record_id = Column(Integer, unique=True, index=True, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    temperature = Column(Float)
    vibration = Column(Float)
    humidity = Column(Float)
    pressure = Column(Float)
    energy_consumption = Column(Float)
    machine_status = Column(Integer)
    prediccion_clase = Column(Integer)
    tipo_falla_predicha = Column(String)
    falla_real_confirmada = Column(String)
    estado_motor = Column(String)
    confianza = Column(Float)
    componentes_afectados = Column(String)

Base.metadata.create_all(bind=engine)
with engine.begin() as connection:
    sensor_log_columns = {column["name"] for column in inspect(engine).get_columns("historial_sensores")}
    missing_columns = {
        "asset_id": "INTEGER",
        "backend_record_id": "INTEGER",
        "tipo_falla_predicha": "VARCHAR",
        "falla_real_confirmada": "VARCHAR",
    }
    for column_name, column_type in missing_columns.items():
        if column_name not in sensor_log_columns:
            connection.execute(text(
                f"ALTER TABLE historial_sensores ADD COLUMN {column_name} {column_type}"
            ))
    connection.execute(text(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_historial_sensores_backend_record_id "
        "ON historial_sensores (backend_record_id)"
    ))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- CONFIGURACIÓN DE FASTAPI ---
app = FastAPI(
    title="AgroSentinel Predictive API",
    description="API para monitoreo predictivo multiclase, diagnóstico inteligente y persistencia de datos",
    version="4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el nuevo modelo multiclase
modelo = joblib.load('data/modelo_rf_final.pkl')

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
    falla_real_confirmada: str = Field(..., min_length=1, max_length=120)
    machine_id: str | None = None
    timestamp: datetime | None = None

def guardar_en_backend(registro: dict) -> int:
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
            response_body = json.loads(response.read().decode("utf-8"))
            backend_record_id = response_body.get("id")
            if not isinstance(backend_record_id, int):
                raise HTTPException(
                    status_code=502,
                    detail="El backend no devolvió el ID de la predicción guardada.",
                )
            return backend_record_id
    except (HTTPError, URLError, TimeoutError) as error:
        raise HTTPException(
            status_code=502,
            detail="No fue posible guardar la predicción en la base de datos de AgroSentinel.",
        ) from error

@app.get("/")
def home():
    return {"mensaje": "Motor monitoreado por IA Multiclase AgroSentinel con base de datos de historial activa."}

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
    
    # 1. LA IA AHORA DEVUELVE EL TEXTO EXACTO DE LA FALLA
    tipo_falla_ia = str(modelo.predict(entrada_df)[0])
    probabilidad = float(modelo.predict_proba(entrada_df).max())
    
    # 2. EVALUAR EL RESULTADO
    # En el dataset de Kaggle, el estado normal suele llamarse "No Failure" o "Normal"
    estado_seguro = tipo_falla_ia.strip().lower() in ["normal", "no failure", "none"]
    
    if estado_seguro:
        estado = "Motor Operando con Normalidad"
        prediccion_num = 0  # 0 para la base de datos
        componentes_texto = "Ninguno (Parámetros estables)"
    else:
        estado = f"¡Alerta! Riesgo Detectado: {tipo_falla_ia}"
        prediccion_num = 1  # 1 para indicar que SÍ hay una falla (para no romper tu DB)
        componentes_texto = f"Diagnóstico IA: {tipo_falla_ia} (Confianza: {probabilidad*100:.1f}%)"

    timestamp = datetime.now(timezone.utc).replace(tzinfo=None)

    # 3. GUARDAR EN BACKEND JAVA/SPRING BOOT
    backend_record_id = guardar_en_backend({
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

    # 4. GUARDAR EN BASE DE DATOS LOCAL SQLITE
    nuevo_registro = SensorLogDB(
        asset_id=datos.asset_id,
        machine_id=datos.machine_id,
        backend_record_id=backend_record_id,
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

    # 5. RETORNAR RESPUESTA AL FRONTEND
    return {
        "id_registro_historial": nuevo_registro.id,
        "asset_id": datos.asset_id,
        "machine_id": nuevo_registro.machine_id,
        "timestamp": nuevo_registro.timestamp,
        "prediccion_clase": prediccion_num,
        "estado_motor": estado,
        "confianza": probabilidad,
        "componentes_afectados": [componentes_texto]
    }

@app.get("/historial")
def ver_historial(db: Session = Depends(get_db)):
    registros = db.query(SensorLogDB).all()
    return {
        "total_registros": len(registros),
        "datos": registros
    }

@app.put("/historial/{id}/feedback")
def guardar_feedback(
    id: int,
    feedback: FeedbackRequest,
    db: Session = Depends(get_db),
    token: str = Header(..., alias="X-Predictive-Token"),
):
    if not PREDICTIVE_API_TOKEN or token != PREDICTIVE_API_TOKEN:
        raise HTTPException(status_code=401, detail="Token del servicio predictivo inválido.")

    registro = db.query(SensorLogDB).filter(
        SensorLogDB.backend_record_id == id
    ).one_or_none()
    if registro is None and feedback.machine_id and feedback.timestamp:
        candidatos = db.query(SensorLogDB).filter(
            SensorLogDB.machine_id == feedback.machine_id,
            SensorLogDB.timestamp == feedback.timestamp,
            SensorLogDB.backend_record_id.is_(None),
        ).limit(2).all()
        if len(candidatos) > 1:
            raise HTTPException(
                status_code=409,
                detail="Hay varias lecturas locales que coinciden con la predicción.",
            )
        if candidatos:
            registro = candidatos[0]
            registro.backend_record_id = id

    if registro is None:
        raise HTTPException(status_code=404, detail="No se encontró la lectura predictiva.")

    falla_confirmada = feedback.falla_real_confirmada.strip()
    clases_validas = {str(clase) for clase in modelo.classes_}
    if falla_confirmada not in clases_validas:
        raise HTTPException(
            status_code=422,
            detail="La falla confirmada no coincide con las clases conocidas por el modelo.",
        )

    registro.falla_real_confirmada = falla_confirmada
    db.commit()
    db.refresh(registro)
    return registro

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