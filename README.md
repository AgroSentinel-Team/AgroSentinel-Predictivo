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
3. **Persistencia Automática (SQLite):** Cada solicitud de inferencia se registra de manera automática en una base de datos local para auditoría y trazabilidad.
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
   pip install fastapi uvicorn pydantic pandas scikit-learn joblib sqlalchemy

---

## 🖥️ Ejecución de la API

Para levantar el servidor de desarrollo en local con recarga en vivo apuntando a la nueva estructura:

python -m uvicorn src.api.main:app --reload

Una vez iniciado, abre tu navegador y accede a la documentación interactiva (Swagger UI):
👉 http://127.0.0.1:8000/docs

---

## 📡 Endpoints Principales

* **`GET /`** -> Mensaje de bienvenida y estado del servicio.
* **`POST /predecir`** -> Recibe las métricas de los sensores, evalúa el modelo, genera el diagnóstico por componentes y **guarda automáticamente el registro en SQLite**.
* **`GET /historial`** -> Retorna el listado completo de todas las telemetrías y predicciones almacenadas en la base de datos.

---

## 🧪 Ejemplo de Payload para Pruebas (`POST /predecir`)

Puedes usar este JSON de ejemplo en Swagger para simular una alerta crítica por sobrecalentamiento:

{
  "temperature": 110.2,
  "vibration": 65.0,
  "humidity": 60.0,
  "pressure": 4.0,
  "energy_consumption": 3.8,
  "machine_status": 1
}
* **Pandas & NumPy** (Manipulación y estructuración de datos)
* **SQLAlchemy & SQLite** (ORM y persistencia relacional)
