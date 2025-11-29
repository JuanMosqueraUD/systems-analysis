# Implementación del Proyecto

## Estructura del Código

```
src/
├── utils.py                    # Utilidades básicas
├── data_processing.py          # Pipeline de datos
├── feature_engineering.py      # PCA y selección
├── classification_engine.py    # Modelos y evaluación
└── main.py                     # Orquestador del pipeline
```

## Classification Engine

### Modelos Implementados
1. Random Forest
2. XGBoost
3. Multi-Layer Perceptron
4. Logistic Regression

### Funcionalidades
- Entrenamiento automático de modelos
- Cross-validation estratificado
- Evaluación con log loss y accuracy
- Comparación de modelos
- Calibración de probabilidades (Isotonic/Platt)
- Ensemble ponderado
- Guardado/carga de modelos
- Generación de submission.csv

## Data Processing

- Carga de train.csv y test.csv
- Validación de datos
- Limpieza automática
- Label encoding
- StandardScaler
- Splits estratificados
- Pipeline unificado

## Feature Engineering

- PCA (opcional)
- Variance threshold
- Configurable desde CLI

## Interfaz de Línea de Comandos

```bash
# Uso básico
python src/main.py

# Con opciones
python src/main.py --pca --n-components 50
python src/main.py --no-calibration
python src/main.py --no-ensemble
python src/main.py --models xgboost mlp
python src/main.py --load-model models/final_model.pkl
python src/main.py --help
```

## Archivos de Salida

```
outputs/
├── submission.csv
├── training_summary.json
└── logs/
    └── pipeline.log

models/
├── final_model.pkl
└── best_single_model.pkl
```

## Ejecución del Pipeline

```
STEP 1: DATA LOADING AND PREPROCESSING
STEP 2: FEATURE ENGINEERING (opcional)
STEP 3: MODEL TRAINING
STEP 4: MODEL EVALUATION
STEP 5: CALIBRATION AND ENSEMBLE
STEP 6: PREDICTION GENERATION
```

## Módulos Principales

### utils.py
- setup_logger(): Configuración de logging
- save_json(), load_json(): Manejo de JSON
- ensure_dir(): Creación de directorios
- get_timestamp(): Timestamps

### data_processing.py
- load_train_data(), load_test_data(): Carga de datos
- validate_data(): Validación
- scale_features(): Normalización
- create_train_val_split(): Splits estratificados

### feature_engineering.py
- fit_transform(): PCA en train
- transform(): PCA en test
- get_pca_info(): Información de varianza

### classification_engine.py
- ModelTrainer: Entrenamiento de modelos
- ModelEvaluator: Evaluación y comparación
- ModelCalibrator: Calibración de probabilidades
- EnsembleBuilder: Creación de ensembles
- ClassificationEngine: Orquestador principal

### main.py
- OttoClassificationPipeline: Pipeline completo
- parse_arguments(): CLI con argparse
- main(): Punto de entrada
python src/main.py --pca --n-components 30 --val-size 0.1




