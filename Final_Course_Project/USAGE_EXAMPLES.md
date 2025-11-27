# Guía de Uso - Otto Classification

## Índice
- [Uso Básico](#uso-básico)
- [Entrenar Modelos Específicos](#entrenar-modelos-específicos)
- [Usar Modelos Guardados](#usar-modelos-guardados)
- [Directorios Personalizados](#directorios-personalizados)
- [Configuraciones Avanzadas](#configuraciones-avanzadas)
- [Entender las Predicciones](#entender-las-predicciones)

## Uso Básico

### Configuración Por Defecto
```bash
python src/main.py
```

Entrena 4 modelos (Random Forest, XGBoost, MLP, Logistic Regression), calibra probabilidades, crea ensemble ponderado y genera `outputs/submission.csv`.

Submission generado con: ENSEMBLE (fusión de modelos calibrados)

## Entrenar Modelos Específicos

### Solo Random Forest
```bash
python src/main.py --models random_forest
```

### Solo XGBoost
```bash
python src/main.py --models xgboost
```

### Dos Modelos
```bash
python src/main.py --models random_forest xgboost
```

### Los 3 Mejores
```bash
python src/main.py --models random_forest xgboost mlp
```

## Usar Modelos Guardados

### Cargar Modelo Sin Reentrenar
```bash
python src/main.py --load-model models/final_model.pkl
```

Genera `outputs/submission_from_loaded_model.csv` sin entrenar.

### Cargar Mejor Modelo Individual
```bash
python src/main.py --load-model models/best_single_model.pkl
```

### Usar Modelo de Otro Experimento
```bash
python src/main.py --load-model experiment1/models/final_model.pkl --output-dir experiment1_retest
```

## Directorios Personalizados

### Cambiar Directorio de Salida
```bash
python src/main.py --output-dir experiment_1
```

### Cambiar Directorio de Modelos
```bash
python src/main.py --model-dir trained_models
```

### Experimento Completo Organizado
```bash
python src/main.py --models xgboost mlp --output-dir exp1_xgb_mlp --model-dir exp1_models
```

## Configuraciones Avanzadas

### Sin Calibración
```bash
python src/main.py --no-calibration
```

### Sin Ensemble
```bash
python src/main.py --no-ensemble
```

Usa solo el mejor modelo individual calibrado.

### Con PCA
```bash
python src/main.py --pca --n-components 50
```

Reduce 93 features a 50 componentes PCA.

### Calibración Sigmoid
```bash
python src/main.py --calibration-method sigmoid
```

Usa Platt Scaling en lugar de Isotonic (default).

### Sin Guardar Modelos
```bash
python src/main.py --no-save
```

### Configuración Rápida
```bash
python src/main.py --models xgboost --no-calibration --no-ensemble --no-save
```

### Configuración Óptima
```bash
python src/main.py --models random_forest xgboost mlp --calibration-method isotonic --output-dir final_submission
```

### Múltiples Experimentos
```bash
# Experimento 1: Todos los modelos
python src/main.py --output-dir exp1_all --model-dir exp1_models

# Experimento 2: Solo XGBoost
python src/main.py --models xgboost --output-dir exp2_xgb --model-dir exp2_models

# Experimento 3: Con PCA
python src/main.py --pca --n-components 30 --output-dir exp3_pca --model-dir exp3_models

# Experimento 4: Sin calibración
python src/main.py --no-calibration --output-dir exp4_nocal --model-dir exp4_models
```

## Entender las Predicciones

### Modelo Usado en Submission

| Comando | Modelo Usado |
|---------|--------------|
| `python src/main.py` | ENSEMBLE (todos calibrados) |
| `--models xgboost` | XGBoost calibrado |
| `--models rf xgboost` | ENSEMBLE (RF + XGBoost) |
| `--no-calibration` | ENSEMBLE sin calibrar |
| `--no-ensemble` | Mejor modelo individual calibrado |
| `--load-model models/final_model.pkl` | El modelo cargado |

### Verificar Modelo Usado

En `outputs/logs/pipeline.log`:
```
Generando predicciones con: ENSEMBLE (fusión de modelos calibrados)
```

En `outputs/training_summary.json`:
```json
{
  "best_model_name": "xgboost",
  "use_calibration": true,
  "use_ensemble": true
}
```

- `use_ensemble: true` = Submission con ENSEMBLE
- `use_ensemble: false` = Submission con best_model_name

## Comandos Rápidos

```bash
# Básico
python src/main.py

# Rápido - Solo XGBoost
python src/main.py --models xgboost

# Óptimo - Los 3 mejores
python src/main.py --models random_forest xgboost mlp

# Testing rápido
python src/main.py --models xgboost --no-calibration --no-save

# Cargar modelo
python src/main.py --load-model models/final_model.pkl

# Experimentar
python src/main.py --models xgboost mlp --output-dir exp1 --model-dir exp1_models

# Ayuda
python src/main.py --help
```


