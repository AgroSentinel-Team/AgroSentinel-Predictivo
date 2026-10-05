# 🌾 AgroSentinel: Sistema de Mantenimiento Predictivo para Motores Agrícolas

**AgroSentinel** es una solución robusta de Machine Learning e ingeniería de backend diseñada para el monitoreo predictivo en tiempo real de motores agrícolas. El sistema combina un modelo de clasificación optimizado (Random Forest), validación estricta de datos, diagnóstico inteligente de fallas y persistencia local mediante base de datos relacional.

---

## 📂 Arquitectura del Proyecto

El proyecto está estructurado bajo estándares profesionales de desarrollo de software y MLOps:

```text
AgroSentinel-Predictivo/
│
├── data/                               # Artefactos de datos, modelos entrenados y DB
│   ├── smart_manufacturing_data.csv    # Dataset base original
│   ├── modelo_rf_final.pkl             # Modelo de Machine Learning entrenado
│   └── agrosentinel_historial.db       # Base de datos SQLite (Persistencia de logs)
│
├── notebooks/                          # Análisis exploratorio y preprocesamiento
│   └── limpieza_datos.ipynb            # Limpieza y preparación del dataset
│
├── src/                                # Código fuente de la aplicación
│   ├── api/                            
│   │   └── main.py                     # Servidor FastAPI (Endpoints de predicción y logs)
│   ├── models/                         
│   │   ├── entrenar.py                 # Script de entrenamiento base del modelo
│   │   └── reentrenar_combinado.py     # Script MLOps (Fusión CSV + SQLite para reentrenamiento)
│   └── utils/                          
│       └── limites.py                  # Definiciones auxiliares y umbrales de sensores
│
├── venv/                               # Entorno virtual de Python
├── .gitignore                          # Archivos excluidos del control de versiones
├── Pruebas.txt                         # Registro de pruebas manuales y payloads de ejemplo
└── README.md                           # Documentación oficial del proyecto
```
---

## 🚀 Características Principales

1. **Predicción con Machine Learning:** Clasificación basada en *Random Forest* entrenado sobre variables físicas clave.
2. **Diagnóstico Inteligente (Reglas de Negocio):** Capa analítica acoplada que traduce la predicción en un informe detallado indicando exactamente qué componente está fallando (temperatura, vibración, presión, etc.).
3. **Persistencia de lecturas:** Cada inferencia se guarda en SQLite para el reentrenamiento local y en PostgreSQL, asociada al activo de AgroSentinel, para consulta desde el panel y la hoja de vida predictiva.
4. **MLOps Híbrido:** Script de reentrenamiento programado para unificar el dataset histórico original con los nuevos registros acumulados en producción mediante `pd.concat`.
5. **Validación Estricta:** Uso de Pydantic (`Field` constraints) para blindar la API ante valores erróneos o estados de máquina inválidos.

---

## 🛠️ Tecnologías Utilizadas

* **Python 3.10+**
* **FastAPI & Uvicorn** (Framework web y servidor ASGI de alto rendimiento)
* **Scikit-Learn & Joblib** (Entrenamiento y serialización del modelo de IA)
* **Pandas & NumPy** (Manipulación y estructuración de datos)
* **SQLAlchemy & SQLite** (ORM y persistencia relacional)

---

## 📥 Instalación y Configuración

Sigue estos pasos para clonar y poner en marcha el proyecto localmente:

1. **Clona el repositorio:**
   git clone <url-de-tu-repositorio>
   cd AgroSentinel-Predictivo

2. **Crea y activa el entorno virtual:**
   python -m venv venv
   # En Windows (PowerShell):
   .\venv\Scripts\Activate

3. **Instala las dependencias necesarias:**
   pip install fastapi uvicorn pydantic pandas numpy scikit-learn joblib sqlalchemy seaborn matplotlib imbalanced-learn xgboost

---

## 🖥️ Ejecución de la API

Para levantar el servidor de desarrollo en local con recarga en vivo apuntando a la nueva estructura:

python -m uvicorn src.api.main:app --reload

Una vez iniciado, abre tu navegador y accede a la documentación interactiva (Swagger UI):
👉 http://127.0.0.1:8000/docs

---

## 📡 Endpoints Principales

* **`GET /`** -> Mensaje de bienvenida y estado del servicio.
* **`POST /predecir`** -> Recibe las métricas de los sensores, evalúa el modelo, genera el diagnóstico por componentes y guarda el registro en SQLite y PostgreSQL, asociado al activo.
* **`GET /historial`** -> Retorna el listado completo de todas las telemetrías y predicciones almacenadas en la base de datos.
* **`PUT /historial/{id}/feedback`** -> Guarda la falla confirmada por el técnico en `falla_real_confirmada`. El `id` de esta ruta es el ID del registro en Spring Boot; el servicio lo relaciona con SQLite mediante `backend_record_id`.

Las predicciones también se envían a `POST http://localhost:8080/api/predictive/records`. El backend valida que el activo exista, tenga motor y coincida con su ID predictivo antes de guardarlas en PostgreSQL. La interfaz React envía feedback a `PUT http://localhost:8080/api/predictive/records/{id}/feedback`; Spring lo persiste en PostgreSQL y lo sincroniza con FastAPI/SQLite. Para producción, define el mismo secreto en `PREDICTIVE_API_TOKEN` para el backend y FastAPI; el valor predeterminado solo es apropiado para desarrollo. Si FastAPI corre en otro contenedor, configura `AGROSENTINEL_BACKEND_URL` con el nombre de host del servicio backend y `PREDICTIVE_API_URL` en Spring Boot con el nombre de host de FastAPI.

El cuerpo del feedback acepta las etiquetas de clase reales del modelo, por ejemplo:

```json
{
  "fallaRealConfirmada": "Overheating"
}
```

---

## 🧪 Ejemplo de Payload para Pruebas (`POST /predecir`)

Puedes usar este JSON de ejemplo en Swagger para simular una alerta crítica por sobrecalentamiento:

{
   "asset_id": 12,
   "machine_id": "MOTOR-12",
  "temperature": 110.2,
  "vibration": 65.0,
  "humidity": 60.0,
  "pressure": 4.0,
  "energy_consumption": 3.8,
  "machine_status": 1
}

El `asset_id` se obtiene del activo registrado y el `machine_id` es el ID predictivo mostrado en su ficha. Ambos son obligatorios y deben corresponder al mismo activo con motor.
* **Pandas & NumPy** (Manipulación y estructuración de datos)
* **SQLAlchemy & SQLite** (ORM y persistencia relacional)
