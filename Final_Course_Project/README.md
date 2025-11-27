# Otto Group Product Classification

Sistema de clasificación multi-clase para 9 categorías de productos usando 93 características numéricas.

Métrica principal: Multi-class logarithmic loss (logloss)

## Estructura del Proyecto

```
Final_Course_Project/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sampleSubmission.csv
├── src/
│   ├── utils.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── classification_engine.py
│   └── main.py
├── models/
├── outputs/
│   ├── logs/
│   └── submission.csv
└── requirements.txt
```

## Funcionalidades

### Procesamiento de Datos
- Carga y validación de datos
- StandardScaler para normalización
- Splits estratificados (train/val)

### Feature Engineering
- PCA para reducción de dimensionalidad (opcional)
- Filtrado por varianza

### Modelos Implementados
- Random Forest
- XGBoost
- Multi-Layer Perceptron
- Logistic Regression

### Classification Engine
- Entrenamiento automático de múltiples modelos
- Evaluación con log loss y accuracy
- Calibración de probabilidades (Isotonic/Platt)
- Ensemble ponderado basado en rendimiento
- Guardado/carga de modelos

### Pipeline Automatizado
1. Carga y preprocesamiento de datos
2. Feature engineering (opcional)
3. Entrenamiento de modelos
4. Evaluación y comparación
5. Calibración de probabilidades
6. Creación de ensemble
7. Generación de predicciones y submission.csv

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### Ejecución Básica
```bash
python src/main.py
```

### Opciones de Configuración

```bash
# Con PCA
python src/main.py --pca --n-components 50

# Sin calibración
python src/main.py --no-calibration

# Sin ensemble
python src/main.py --no-ensemble

# Modelos específicos
python src/main.py --models random_forest xgboost

# Cargar modelo guardado
python src/main.py --load-model models/final_model.pkl

# Directorios personalizados
python src/main.py --output-dir exp1 --model-dir exp1_models

# Ver todas las opciones
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

### Métricas Reportadas

- Log Loss (métrica principal)
- Accuracy
- Confusion Matrix
- Mejora por calibración
- Pesos del ensemble

## Arquitectura

### Principios de Diseño
- Arquitectura modular con 5 componentes principales
- Stack técnico: NumPy, Pandas, Scikit-learn, XGBoost
- Interfaces claras entre componentes
- Pipeline reproducible
- Validación estratificada
- Calibración de probabilidades
- Logging integrado

## Equipo

- **Juan Diego Lozada** (20222020014) - Analista de Sistemas
- **Juan Pablo Mosquera** (20221020026) - Desarrollador
- **María Alejandra Ortiz Sánchez** (20242020223) - Ingeniera de Calidad
- **Jeison Felipe Cuenca** (20242020043) - Gerente de Proyecto

## Referencias

- [Otto Group Product Classification Challenge (Kaggle)](https://www.kaggle.com/c/otto-group-product-classification-challenge)
- Workshops 1-4 del curso de Análisis de Sistemas
- Scikit-learn Documentation
- XGBoost Documentation

## Notas Técnicas

### Sobre el Log Loss

La métrica principal del challenge es multi-class logloss:

```
logloss = -1/N * Σ Σ y_ij * log(p_ij)
```

Donde:
- y_ij = 1 si la muestra i pertenece a la clase j, 0 en otro caso
- p_ij = probabilidad predicha de que la muestra i pertenezca a la clase j

**Valores más bajos son mejores**. El log loss penaliza fuertemente las predicciones muy confiadas pero incorrectas.

### Sobre la Calibración

La calibración ajusta las probabilidades predichas para que reflejen mejor la verdadera likelihood. Dos métodos implementados:

- **Platt Scaling (sigmoid)**: Ajusta con regresión logística
- **Isotonic Regression**: Ajusta con función monotónica no paramétrica (más flexible)

La calibración es especialmente importante para log loss.

### Sobre el Ensemble

El ensemble usa **soft voting** con pesos calculados inversamente proporcionales al log loss de validación:

```
weight_i = 1 / logloss_i
```

Normalizado para que sumen 1. Modelos con menor log loss tienen mayor peso.
