# 🌾 AgroSentinel: Sistema de Mantenimiento Predictivo para Motores Agrícolas

**AgroSentinel** es una solución robusta de Machine Learning e ingeniería de backend diseñada para el monitoreo predictivo en tiempo real de motores agrícolas. La API combina reglas físicas, votación mayoritaria de Random Forest, SVM y XGBoost, y el contexto temporal de una red Long Short-Term Memory (LSTM), además de validación estricta y persistencia local relacional.

---

## 📂 Arquitectura del Proyecto

El proyecto está estructurado bajo estándares profesionales de desarrollo de software y MLOps:

```text
AgroSentinel-Predictivo/
│
├── data/                               # Artefactos de datos, modelos entrenados y DB
│   ├── smart_manufacturing_data.csv    # Dataset base original
│   ├── modelo_rf.pkl                    # Random Forest
│   ├── modelo_svm.pkl                   # SVM RBF
│   ├── modelo_xgb.pkl                   # XGBoost y LabelEncoder
│   ├── modelo_lstm_final.pt            # Red LSTM entrenada
│   └── agrosentinel_historial.db       # Base de datos SQLite (Persistencia de logs)
│
├── notebooks/                          # Análisis exploratorio y preprocesamiento
│   └── limpieza_datos.ipynb            # Limpieza y preparación del dataset
│
├── src/                                # Código fuente de la aplicación
│   ├── api/                            
│   │   └── main.py                     # Servidor FastAPI (Endpoints de predicción y logs)
│   ├── models/                         
│   │   ├── entrenar_lstm.py            # Entrenamiento temporal de la LSTM
│   │   ├── entrenar_tabular.py         # Entrenamiento del ensamble tabular
│   │   └── reentrena.py                # Reentrenamiento LSTM con feedback técnico
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

1. **Predicción híbrida:** Las reglas físicas tienen prioridad; si no detectan una falla, tres modelos tabulares votan el diagnóstico estático. La LSTM analiza hasta diez lecturas para aportar la tendencia temporal.
2. **Diagnóstico Inteligente (Reglas de Negocio):** Las reglas y los votos se traducen en un informe detallado indicando qué componente podría estar fallando (temperatura, vibración, presión, etc.).
3. **Persistencia de lecturas:** Cada inferencia se guarda en SQLite para el reentrenamiento local y en PostgreSQL, asociada al activo de AgroSentinel, para consulta desde el panel y la hoja de vida predictiva.
4. **MLOps Híbrido:** Script de reentrenamiento programado para unificar el dataset histórico original con los nuevos registros acumulados en producción mediante `pd.concat`.
5. **Validación Estricta:** Uso de Pydantic (`Field` constraints) para blindar la API ante valores erróneos o estados de máquina inválidos.

---

## 🛠️ Tecnologías Utilizadas

* **Python 3.10+**
* **FastAPI & Uvicorn** (Framework web y servidor ASGI de alto rendimiento)
* **PyTorch** (Entrenamiento e inferencia de la red LSTM)
* **Scikit-Learn, imbalanced-learn y XGBoost** (Modelos tabulares, métricas y balanceo SMOTE)
* **Joblib** (Serialización de los modelos tabulares)
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
   pip install -r requirements.txt

---

## 🖥️ Ejecución de la API

Para levantar el servidor de desarrollo en local con recarga en vivo apuntando a la nueva estructura:

python -m uvicorn src.api.main:app --reload

Una vez iniciado, abre tu navegador y accede a la documentación interactiva (Swagger UI):
👉 http://127.0.0.1:8000/docs

Antes de iniciar la API, genera los tres modelos tabulares con `python -m src.models.entrenar_tabular` y la LSTM con `python -m src.models.entrenar_lstm`. La API los carga en memoria al arrancar y requiere los cuatro artefactos en `data/`.

---

## 📡 Endpoints Principales

* **`GET /`** -> Mensaje de bienvenida y estado del servicio.
* **`POST /predecir`** -> Prioriza las reglas físicas; si no se activa ninguna, combina los votos tabulares (RF resuelve un empate triple). La LSTM devuelve la tendencia de las últimas diez lecturas. Guarda el veredicto en SQLite y PostgreSQL y responde con `tipoFallaPredicha`, `detalle_consenso` y `tendencia_lstm`.
* **`GET /historial`** -> Retorna el listado completo de todas las telemetrías y predicciones almacenadas en la base de datos.
* **`PUT /historial/{id}/feedback`** -> Guarda la falla confirmada por el técnico en `falla_real_confirmada`. El `id` de esta ruta es el ID del registro en Spring Boot; el servicio lo relaciona con SQLite mediante `backend_record_id`.
* **`GET /api/predictive/validation/summary`** (Spring Boot) -> Retorna la cantidad de lecturas confirmadas por técnicos, los aciertos y la exactitud acumulada del modelo.

Las predicciones también se envían a `POST http://localhost:8080/api/predictive/records` con el encabezado `X-Predictive-Token` y el payload `application/json`. El backend valida que el activo exista, tenga motor y coincida con su ID predictivo antes de guardarlas en PostgreSQL. La interfaz React envía feedback a `PUT http://localhost:8080/api/predictive/records/{id}/feedback`; Spring lo persiste en PostgreSQL y lo sincroniza con FastAPI/SQLite. Para producción, configura el mismo secreto en `PREDICTIVE_API_TOKEN` para Spring Boot y `PREDICTIVE_TOKEN` para FastAPI. FastAPI también acepta el nombre anterior `PREDICTIVE_API_TOKEN` como alternativa de compatibilidad; los valores predeterminados son solo para desarrollo. Si FastAPI corre en otro contenedor, configura `AGROSENTINEL_BACKEND_URL` con el nombre de host del servicio backend y `PREDICTIVE_API_URL` en Spring Boot con el nombre de host de FastAPI.

La confianza de la predicción refleja la proporción de votos tabulares para el veredicto (o 1.0 cuando prevalece una regla física); `confianza_lstm` corresponde a la tendencia temporal. Ninguna es una garantía ni una confirmación de acierto. La exactitud en vivo se calcula comparando `tipoFallaPredicha` con `fallaRealConfirmada` (sin distinguir mayúsculas/minúsculas) y solo incluye lecturas con feedback técnico. Las lecturas nuevas quedan pendientes de esa validación real.

Para entrenar la LSTM desde la raíz del proyecto, ejecuta `python -m src.models.entrenar_lstm`. El entrenamiento ordena las lecturas por máquina y fecha, usa ventanas deslizantes de diez lecturas y reserva cronológicamente el último 20 % de cada máquina para validación. Genera las etiquetas objetivo con las mismas reglas físicas del servicio, pondera las clases minoritarias y guarda el modelo y sus metadatos en `data/modelo_lstm_final.pt`. La validación mide qué tan bien aprende esas reglas, no exactitud frente a fallas reales independientes; revisa el F1 y el reporte por clase además de la exactitud global.

Para entrenar los modelos tabulares del ensamble híbrido, ejecuta `python -m src.models.entrenar_tabular`. El script aplica las reglas físicas para generar las etiquetas, balancea las clases con SMOTE y entrena Random Forest, SVM con kernel RBF y XGBoost. RF y SVM conservan las etiquetas de texto; XGBoost usa etiquetas codificadas y guarda el `LabelEncoder` junto al estimador para recuperar las clases originales. Para mantener práctico el ajuste del SVM RBF con el dataset balanceado, se seleccionan hasta 2.000 ejemplos por clase; RF y XGBoost usan todos los ejemplos generados por SMOTE. Los artefactos se guardan por separado en `data/modelo_rf.pkl`, `data/modelo_svm.pkl` y `data/modelo_xgb.pkl`.

En producción, cada nueva lectura se combina con hasta las nueve lecturas anteriores de esa máquina almacenadas en SQLite. Si aún no hay diez lecturas, la ventana se completa repitiendo la primera lectura disponible. Para reentrenar con las confirmaciones técnicas, ejecuta `python -m src.models.reentrena`; las etiquetas confirmadas se combinan con el histórico y vuelven a validarse cronológicamente. La exactitud global puede ocultar errores en las clases raras; consulta el F1 y el reporte por clase que imprime el entrenamiento.

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
