# ML Challenge - Titanic Classifier

## Descripción
Clasificador binario para predecir la supervivencia de pasajeros basado en el dataset del Titanic. Incluye un pipeline automatizado de entrenamiento, una API para inferencias en tiempo real, y flujos CI/CD para garantizar calidad y reproducibilidad.

---

## Estructura del proyecto

```
.
├── data/                  # Dataset original (train.csv - no incluido)
├── models/                # Modelos entrenados (.pkl)
├── src/
│   ├── training/          # Preprocesamiento y entrenamiento
│   └── api/               # API en FastAPI
├── tests/                 # Pruebas unitarias con Pytest
├── .github/workflows/     # Workflows de CI/CD (GitHub Actions)
├── requirements.txt       # Dependencias del proyecto
└── Dockerfile             # Imagen Docker para la API
```

---

## Instalación y ejecución

### Requisitos
- Python 3.10+
- pip
- (Opcional) Docker

### Instalación
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Entrenamiento de modelos A y B
```bash
python src/training/pipeline.py
```

### Ejecutar la API
```bash
uvicorn src.api.main:app --reload
```

Accede a la documentación Swagger en: http://localhost:8000/docs

---

## Ejemplo de predicción

### Con Swagger UI
Usa el siguiente JSON:
```json
[
  {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22.0,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
  }
]
```

### Con curl
```bash
curl -X POST http://localhost:8000/predict \
  -H "x-api-key: rappi-secret" \
  -H "Content-Type: application/json" \
  -d '[{"Pclass":3,"Sex":"male","Age":22,"SibSp":1,"Parch":0,"Fare":7.25,"Embarked":"S"}]'
```

---

## Evaluación del modelo

### Modelo A (Logistic Regression)
```
F1-score: 0.80
Accuracy: 0.80
```

### Modelo B (Gradient Boosting + Preprocesamiento)
```
F1-score: 0.82
Accuracy: 0.83
```

Modelo B supera ligeramente al baseline (Modelo A), logrando mejor balance entre precisión y recall.

---

## Features más importantes

- Sex_male: ser hombre reduce la probabilidad de supervivencia.
- Pclass_1 y Fare: pasajeros en primera clase y con mayores tarifas tienden a sobrevivir más.
- Embarked_S: el puerto de embarque también influye en el resultado.

---

## Pruebas automáticas

- Pruebas de entrenamiento y predicción incluidas en `tests/`
- Ejecutar:
```bash
pytest tests/
```

---

## CI/CD y despliegue

### CI (GitHub Actions)
- Se ejecuta con cada push o pull request a `main`
- Corre tests automáticamente con `pytest`

### CD (opcional)
- Soporte para Google Cloud Run (requiere secretos en GitHub)

### Docker
```bash
docker build -t titanic-api .
docker run -p 8080:8080 titanic-api
```

---

## Checklist de funcionalidades

- [x] Clasificador binario con dos versiones (A: Logistic, B: Boosting)
- [x] Pipeline automático (`pipeline.py`) con logging y profiling
- [x] API RESTful con FastAPI
- [x] Validación de entrada con Pydantic
- [x] Logs de errores y rendimiento
- [x] Seguridad por API-Key (`x-api-key`)
- [x] Endpoint `/feature-importances`
- [x] A/B Testing para comparar modelos
- [x] Tests unitarios (Pytest)
- [x] CI/CD con GitHub Actions
- [x] Contenerización con Docker

---

## Conclusión

### Métricas de evaluación
El modelo B (mejorado) alcanzó un F1-score de 0.82, superando al modelo base (F1: 0.80). Ambas versiones muestran desempeño robusto, considerando el desbalance del dataset.

### Features clave
La variable más influyente es el género (Sex), seguida por clase de pasajero (Pclass) y tarifa pagada (Fare). También destacan variables como Embarked_S y Parch.

### Producción y MLOps

- Entrenamiento desacoplado del consumo.
- Serialización con joblib y carga dinámica en la API.
- Despliegue portable con Docker y preparado para escalar en servicios como Google Cloud Run.
- Arquitectura modular basada en buenas prácticas de MLOps.

---

## Autor

Cristian Arbelaez  
GitHub: https://github.com/cdarbelaez
