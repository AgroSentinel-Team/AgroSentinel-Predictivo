from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request as FastAPIRequest
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, func, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from src.models.lstm_model import load_lstm_model, predict_lstm

# --- CONFIGURACIÓN DE LA BASE DE DATOS SQLITE ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_URL = f"sqlite:///{(DATA_DIR / 'agrosentinel_historial.db').as_posix()}"
FEATURE_COLUMNS = [
    "temperature",
    "vibration",
    "humidity",
    "pressure",
    "energy_consumption",
    "machine_status",
]
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
def cargar_modelos():
    def cargar_modelo_tabular(nombre_archivo):
        ruta_modelo = DATA_DIR / nombre_archivo
        if not ruta_modelo.is_file():
            raise FileNotFoundError(
                f"No se encontró '{ruta_modelo}'. Entrena los modelos tabulares "
                "con 'python -m src.models.entrenar_tabular'."
            )
        return joblib.load(ruta_modelo)

    modelos = {
        "rf": cargar_modelo_tabular("modelo_rf.pkl"),
        "svm": cargar_modelo_tabular("modelo_svm.pkl"),
    }
    artefacto_xgb = cargar_modelo_tabular("modelo_xgb.pkl")
    if not isinstance(artefacto_xgb, dict) or not {
        "model",
        "label_encoder",
        "feature_columns",
    }.issubset(artefacto_xgb):
        raise ValueError(
            "El artefacto data/modelo_xgb.pkl debe incluir model, label_encoder y feature_columns."
        )
    if artefacto_xgb["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("El orden de características de XGBoost no coincide con la API.")
    modelos["xgb"] = artefacto_xgb
    modelos["lstm"], modelos["lstm_metadata"] = load_lstm_model(
        DATA_DIR / "modelo_lstm_final.pt"
    )
    return modelos


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.modelos = cargar_modelos()
    yield


app = FastAPI(
    title="AgroSentinel Predictive API",
    description="API MLOps: Predicción Multiclase, Diagnóstico por Sensores y Feedback Loop",
    version="7.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


def diagnosticar_reglas_fisicas(datos: DatosSensores):
    if datos.temperature > 105.0:
        return "Overheating"
    if datos.vibration > 90.0:
        return "Vibration Issue"
    if datos.pressure < 1.5 or datos.pressure > 4.5:
        return "Pressure Drop"
    if datos.energy_consumption > 4.5:
        return "Electrical Fault"
    return "Normal"


def votar_modelos_tabulares(voto_rf, voto_svm, voto_xgb):
    votos = [voto_rf, voto_svm, voto_xgb]
    conteo_votos = {
        etiqueta: votos.count(etiqueta)
        for etiqueta in dict.fromkeys(votos)
    }
    maximo_votos = max(conteo_votos.values())
    ganadores = [
        etiqueta
        for etiqueta, cantidad in conteo_votos.items()
        if cantidad == maximo_votos
    ]
    diagnostico = ganadores[0] if len(ganadores) == 1 else voto_rf
    return diagnostico, maximo_votos / len(votos), conteo_votos


def explicar_confianza(
    voto_reglas,
    tipo_falla_final,
    confianza_final,
    conteo_votos,
    voto_rf,
    voto_svm,
    voto_xgb,
):
    porcentaje = round(confianza_final * 100)
    if voto_reglas != "Normal":
        return (
            f"El {porcentaje}% corresponde a la aplicación de la regla física "
            f"'{tipo_falla_final}' (no a una probabilidad calibrada del modelo). "
            "Los modelos tabulares no votaron porque prevaleció el umbral físico."
        )

    votos_texto = (
        f"Random Forest: {voto_rf}; SVM: {voto_svm}; XGBoost: {voto_xgb}."
    )
    cantidad_mayoritaria = max(conteo_votos.values())
    if cantidad_mayoritaria == 1:
        motivo = (
            "Los tres modelos discrepan (1 de 3 votos para cada clase); "
            f"se eligió '{tipo_falla_final}' porque Random Forest resuelve el empate."
        )
    else:
        motivo = (
            f"{cantidad_mayoritaria} de 3 modelos coinciden en "
            f"'{tipo_falla_final}'."
        )
    return (
        f"El {porcentaje}% es el nivel de acuerdo entre los tres modelos tabulares, "
        f"no una probabilidad calibrada de acierto. {motivo} {votos_texto}"
    )


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
def predecir_falla(
    datos: DatosSensores,
    request: FastAPIRequest,
    db: Session = Depends(get_db),
):
    modelos = request.app.state.modelos
    lectura_actual = {
        "temperature": datos.temperature,
        "vibration": datos.vibration,
        "humidity": datos.humidity,
        "pressure": datos.pressure,
        "energy_consumption": datos.energy_consumption,
        "machine_status": datos.machine_status,
    }

    fila_actual = pd.DataFrame([lectura_actual], columns=FEATURE_COLUMNS)

    # 1. Diagnóstico físico, voto tabular y contexto temporal
    voto_reglas = diagnosticar_reglas_fisicas(datos)
    if voto_reglas == "Normal":
        voto_rf = str(modelos["rf"].predict(fila_actual)[0])
        voto_svm = str(modelos["svm"].predict(fila_actual)[0])
        prediccion_xgb = modelos["xgb"]["model"].predict(fila_actual)
        voto_xgb = str(
            modelos["xgb"]["label_encoder"].inverse_transform(
                np.asarray(prediccion_xgb, dtype=np.int64).reshape(-1)
            )[0]
        )
    else:
        voto_rf = voto_svm = voto_xgb = "No ejecutado (regla física)"

    if voto_reglas != "Normal":
        diagnostico_tabular = None
        tipo_falla_final = voto_reglas
        confianza_final = 1.0
        conteo_votos = {}
    else:
        diagnostico_tabular, confianza_final, conteo_votos = votar_modelos_tabulares(
            voto_rf, voto_svm, voto_xgb
        )
        tipo_falla_final = diagnostico_tabular

    razon_confianza = explicar_confianza(
        voto_reglas,
        tipo_falla_final,
        confianza_final,
        conteo_votos,
        voto_rf,
        voto_svm,
        voto_xgb,
    )

    # 1. Secuencia temporal: últimas lecturas de esta máquina y la lectura recién recibida
    lecturas_previas = (
        db.query(SensorLogDB)
        .filter(SensorLogDB.machine_id == datos.machine_id)
        .order_by(SensorLogDB.timestamp.desc(), SensorLogDB.id.desc())
        .limit(modelos["lstm_metadata"]["sequence_length"] - 1)
        .all()
    )
    secuencia = [
        {
            "temperature": registro.temperature,
            "vibration": registro.vibration,
            "humidity": registro.humidity,
            "pressure": registro.pressure,
            "energy_consumption": registro.energy_consumption,
            "machine_status": registro.machine_status,
        }
        for registro in reversed(lecturas_previas)
    ]
    secuencia.append(lectura_actual)

    tendencia_lstm, confianza_lstm = predict_lstm(
        modelos["lstm"], modelos["lstm_metadata"], secuencia
    )

    # 2. Alertas descriptivas adicionales para el estado y los componentes afectados
    alertas_sensores = diagnosticar_fallas_sensores(datos)
    estado_normal = tipo_falla_final.strip().lower() in ["normal", "no failure", "none"]
    estado = (
        "Motor Operando con Normalidad"
        if estado_normal and not alertas_sensores
        else "¡Alerta! Riesgo Detectado en Motor"
    )
    prediccion_num = 0 if estado_normal else 1

    mensajes_finales = []
    if not estado_normal:
        origen = "Regla física" if voto_reglas != "Normal" else "Votación tabular"
        mensajes_finales.append(
            f"{origen}: {tipo_falla_final} -> {obtener_explicacion_falla_ia(tipo_falla_final)}"
        )
    if alertas_sensores:
        mensajes_finales.append("Sensores Críticos: " + " | ".join(alertas_sensores))
    componentes_texto = (
        " || ".join(mensajes_finales)
        if mensajes_finales
        else "Ninguno (Parámetros estables)"
    )

    timestamp = datetime.now(timezone.utc).replace(tzinfo=None)

    detalle_consenso = {
        "random_forest": voto_rf,
        "svm": voto_svm,
        "xgboost": voto_xgb,
        "reglas_fisicas": voto_reglas,
        "conteo_votos": conteo_votos,
        "diagnostico_tabular": diagnostico_tabular,
        "veredicto_final": tipo_falla_final,
    }

    # 3. GUARDAR EN SPRING BOOT (Y recuperar el ID)
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
        "tipoFallaPredicha": tipo_falla_final,
        "motorState": estado,
        "confidence": confianza_final,
        "affectedComponents": componentes_texto,
        "tendencia_lstm": tendencia_lstm,
        "confianza_lstm": confianza_lstm,
        "razon_confianza": razon_confianza,
    })

    # 4. GUARDAR EN SQLITE LOCAL (Conectando el ID de Spring)
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
        tipo_falla_predicha=tipo_falla_final,
        estado_motor=estado,
        confianza=confianza_final,
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
        "tipoFallaPredicha": tipo_falla_final,
        "tipo_falla_predicha": tipo_falla_final,
        "detalle_consenso": detalle_consenso,
        "tendencia_lstm": tendencia_lstm,
        "confianza_lstm": confianza_lstm,
        "razon_confianza": razon_confianza,
        "estado_motor": estado,
        "confianza": confianza_final,
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