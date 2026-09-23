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
│   │   └── main.py                     # Servidor FastAPI (Endprinópolis y Endpoints)
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
