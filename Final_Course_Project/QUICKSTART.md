# Quick Start Guide

## Instalación

```bash
pip install numpy pandas scikit-learn xgboost matplotlib
```

## Verificar Datos

Asegúrate de tener los archivos en `data/`:
- `train.csv`
- `test.csv`
- `sampleSubmission.csv`

## Ejecución

```bash
python src/main.py
```

## Configuraciones Comunes

```bash
# Con PCA (reduce dimensionalidad)
python src/main.py --pca --n-components 50

# Sin ensemble (solo mejor modelo)
python src/main.py --no-ensemble

# Sin calibración
python src/main.py --no-calibration

# Modelos específicos
python src/main.py --models random_forest xgboost

# Cargar modelo guardado
python src/main.py --load-model models/final_model.pkl

# Directorios personalizados
python src/main.py --output-dir exp1 --model-dir exp1_models
```

## Archivos Generados

```
outputs/
├── submission.csv           # Para subir a Kaggle
├── training_summary.json    # Métricas
└── logs/
    └── pipeline.log

models/
├── final_model.pkl          # Modelo ensemble
└── best_single_model.pkl    # Mejor modelo individual
```

## Ver Resultados

```bash
# Ver resumen
cat outputs/training_summary.json

# Ver log completo
cat outputs/logs/pipeline.log
```

## Formato de Submission

El archivo `submission.csv` tiene el formato:
```
id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9
1,0.1,0.05,0.3,0.2,0.1,0.05,0.1,0.05,0.05
```

Cada fila suma 1.0 (probabilidades normalizadas).

## Ejemplo de Salida del Pipeline

### Inicio
```
======================================================================
OTTO GROUP CLASSIFICATION PIPELINE
======================================================================
Configuration:
  - Use PCA: False
  - Calibration: True
  - Ensemble: True
  - Validation size: 0.2
```

### Carga de Datos
```
======================================================================
STEP 1: DATA LOADING AND PREPROCESSING
======================================================================
Loading training data from data/train.csv
Loaded 61878 samples with 95 columns
```

### Entrenamiento
```
======================================================================
STEP 3: MODEL TRAINING
======================================================================
Training Random Forest
Training XGBoost
Training MLP
Training Logistic Regression
```

### Evaluación
```
======================================================================
STEP 4: MODEL EVALUATION
======================================================================
xgboost - Log Loss: 0.4987, Accuracy: 0.7956
mlp - Log Loss: 0.5123, Accuracy: 0.7843
random_forest - Log Loss: 0.5234, Accuracy: 0.7821
logistic - Log Loss: 0.6234, Accuracy: 0.7521
```

### Calibración y Ensemble
```
======================================================================
STEP 5: CALIBRATION AND ENSEMBLE
======================================================================
Calibrating models using isotonic method...
Creating weighted ensemble...
Calculated weights: {'xgboost': 0.28, 'mlp': 0.27, 'random_forest': 0.27}
```

### Finalización
```
======================================================================
STEP 6: PREDICTION GENERATION
======================================================================
Submission file saved to: outputs/submission.csv
======================================================================
PIPELINE COMPLETED SUCCESSFULLY!
======================================================================
```

## Solución de Problemas

### Error: ModuleNotFoundError xgboost
```bash
pip install xgboost
```

### Error: FileNotFoundError train.csv
Verifica que los archivos estén en `data/`:
```bash
ls data/
```

### Warning: Convergence warning (MLP)
Este warning es normal y puede ignorarse.
