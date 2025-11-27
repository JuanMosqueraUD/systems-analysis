"""
Main entry point for the Otto Group Classification System.

Simplified implementation for university semester project.
Orchestrates: data loading → preprocessing → feature engineering → 
training → evaluation → prediction generation.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    setup_logger, ensure_dir, save_submission, 
    get_data_path, get_models_path, get_outputs_path
)
from data_processing import DataProcessor
from feature_engineering import FeatureEngineer, create_simple_feature_engineer, create_pca_feature_engineer
from classification_engine import ClassificationEngine


def predict_with_loaded_model(model_path: str, output_dir: str = 'outputs'):
    """
    Cargar modelo guardado y generar predicciones sin reentrenar.
    
    Args:
        model_path: Ruta al modelo guardado (.pkl)
        output_dir: Directorio para guardar submission.csv
    """
    from pathlib import Path
    import pickle
    
    logger = setup_logger('LoadedModelPredictor', str(Path(output_dir) / 'logs' / 'predict.log'))
    
    logger.info("=" * 70)
    logger.info("PREDICTION WITH LOADED MODEL")
    logger.info("=" * 70)
    logger.info(f"Model path: {model_path}")
    
    # 1. Cargar modelo
    logger.info("Loading model...")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully: {type(model).__name__}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # 2. Cargar y procesar datos de test
    logger.info("Loading and processing test data...")
    processor = DataProcessor(logger=logger)
    
    data_path = get_data_path()
    test_path = data_path / 'test.csv'
    train_path = data_path / 'train.csv'  # Necesario para fit del scaler
    
    # Procesar train (solo para ajustar el scaler)
    df_train = processor.load_train_data(str(train_path))
    processor.validate_data(df_train)
    df_train = processor.clean_data(df_train)
    X_train, y_train, _ = processor.prepare_features_and_target(df_train, is_train=True)
    X_train_scaled, _ = processor.scale_features(X_train, None, fit=True)
    
    # Procesar test
    X_test, test_ids = processor.process_test_pipeline(str(test_path))
    
    logger.info(f"Test data shape: {X_test.shape}")
    
    # 3. Predecir
    logger.info("Generating predictions...")
    predictions = model.predict_proba(X_test)
    
    # 4. Guardar submission
    output_path = Path(output_dir) / 'submission_from_loaded_model.csv'
    ensure_dir(str(Path(output_dir)))
    save_submission(predictions, test_ids, str(output_path))
    
    logger.info(f"Predictions saved to: {output_path}")
    logger.info(f"Predictions shape: {predictions.shape}")
    logger.info("=" * 70)
    logger.info("PREDICTION COMPLETED!")
    logger.info("=" * 70)


class OttoClassificationPipeline:
    """
    Complete pipeline for Otto Group Product Classification.
    
    Simplified architecture for university project:
    1. Data Loading & Preprocessing (StandardScaler, stratified split)
    2. Feature Engineering (optional PCA)
    3. Model Training (RF, XGBoost, MLP, Logistic)
    4. Model Evaluation (log loss, accuracy)
    5. Probability Calibration (Platt/Isotonic)
    6. Ensemble Creation (weighted voting)
    7. Prediction & Submission Generation
    """
    
    def __init__(self, 
                 use_pca: bool = False,
                 n_components: int = 50,
                 use_calibration: bool = True,
                 calibration_method: str = 'isotonic',
                 use_ensemble: bool = True,
                 validation_size: float = 0.2,
                 random_state: int = 42,
                 models_to_train: list = None,
                 output_dir: str = 'outputs',
                 model_dir: str = 'models'):
        """
        Initialize the classification pipeline.
        
        Args:
            use_pca: Whether to apply PCA for dimensionality reduction
            n_components: Number of PCA components (if use_pca=True)
            use_calibration: Whether to calibrate model probabilities
            calibration_method: Calibration method ('isotonic' or 'sigmoid')
            use_ensemble: Whether to create ensemble from multiple models
            validation_size: Proportion of data for validation
            random_state: Random seed for reproducibility
            models_to_train: List of models to train (default: all)
            output_dir: Directory for outputs
            model_dir: Directory for models
        """
        # Custom directories
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)
        ensure_dir(str(self.output_dir))
        ensure_dir(str(self.model_dir))
        
        # Setup logger
        log_file = self.output_dir / 'logs' / 'pipeline.log'
        self.logger = setup_logger('OttoPipeline', str(log_file))
        
        # Configuration
        self.use_pca = use_pca
        self.n_components = n_components
        self.use_calibration = use_calibration
        self.calibration_method = calibration_method
        self.use_ensemble = use_ensemble
        self.validation_size = validation_size
        self.random_state = random_state
        self.models_to_train = models_to_train if models_to_train else [
            'random_forest', 'xgboost', 'mlp', 'logistic'
        ]
        
        # Initialize components
        self.logger.info("=" * 70)
        self.logger.info("OTTO GROUP CLASSIFICATION PIPELINE")
        self.logger.info("=" * 70)
        self.logger.info(f"Configuration:")
        self.logger.info(f"  - Models to train: {', '.join(self.models_to_train)}")
        self.logger.info(f"  - Use PCA: {use_pca} (n_components={n_components if use_pca else 'N/A'})")
        self.logger.info(f"  - Calibration: {use_calibration} (method={calibration_method})")
        self.logger.info(f"  - Ensemble: {use_ensemble}")
        self.logger.info(f"  - Validation size: {validation_size}")
        self.logger.info(f"  - Random state: {random_state}")
        self.logger.info(f"  - Output directory: {self.output_dir}")
        self.logger.info(f"  - Model directory: {self.model_dir}")
        
        self.data_processor = DataProcessor(logger=self.logger)
        
        if use_pca:
            self.feature_engineer = create_pca_feature_engineer(
                n_components=n_components,
                logger=self.logger
            )
        else:
            self.feature_engineer = create_simple_feature_engineer(logger=self.logger)
        
        self.classification_engine = ClassificationEngine(
            use_calibration=use_calibration,
            calibration_method=calibration_method,
            use_ensemble=use_ensemble,
            logger=self.logger
        )
        
        # Data storage
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.X_test = None
        self.test_ids = None
    
    def load_and_preprocess_data(self):
        """
        Step 1: Load and preprocess training and test data.
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 1: DATA LOADING AND PREPROCESSING")
        self.logger.info("=" * 70)
        
        # Get data paths
        data_path = get_data_path()
        train_path = data_path / 'train.csv'
        test_path = data_path / 'test.csv'
        
        # Process training data
        self.X_train, self.X_val, self.y_train, self.y_val = \
            self.data_processor.process_train_pipeline(
                str(train_path),
                validation_size=self.validation_size
            )
        
        # Process test data
        self.X_test, self.test_ids = self.data_processor.process_test_pipeline(
            str(test_path)
        )
        
        self.logger.info(f"\nData shapes:")
        self.logger.info(f"  - X_train: {self.X_train.shape}")
        self.logger.info(f"  - X_val: {self.X_val.shape}")
        self.logger.info(f"  - X_test: {self.X_test.shape}")
    
    def apply_feature_engineering(self):
        """
        Step 2: Apply feature engineering (PCA, feature selection, etc.).
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 2: FEATURE ENGINEERING")
        self.logger.info("=" * 70)
        
        # Transform training and validation data
        self.X_train, self.X_val = self.feature_engineer.fit_transform(
            self.X_train, self.X_val
        )
        
        # Transform test data
        self.X_test = self.feature_engineer.transform(self.X_test)
        
        self.logger.info(f"\nFeature-engineered shapes:")
        self.logger.info(f"  - X_train: {self.X_train.shape}")
        self.logger.info(f"  - X_val: {self.X_val.shape}")
        self.logger.info(f"  - X_test: {self.X_test.shape}")
        
        # Log feature engineering info
        feature_info = self.feature_engineer.get_feature_info()
        self.logger.info(f"\nFeature engineering summary:")
        for key, value in feature_info.items():
            if key != 'explained_variance_ratio' and key != 'cumulative_variance':
                self.logger.info(f"  - {key}: {value}")
    
    def train_models(self):
        """
        Step 3: Train all classification models.
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 3: MODEL TRAINING")
        self.logger.info("=" * 70)
        
        # Crear configuración solo para modelos seleccionados
        models_config = {}
        
        if 'random_forest' in self.models_to_train:
            models_config['random_forest'] = {
                'n_estimators': 300,
                'max_depth': 20
            }
        
        if 'xgboost' in self.models_to_train:
            models_config['xgboost'] = {
                'n_estimators': 300,
                'learning_rate': 0.1,
                'max_depth': 6
            }
        
        if 'mlp' in self.models_to_train:
            models_config['mlp'] = {
                'hidden_layers': (256, 128),
                'max_iter': 200
            }
        
        if 'logistic' in self.models_to_train:
            models_config['logistic'] = {}
        
        # Train selected models
        models = self.classification_engine.train_all_models(
            self.X_train, self.y_train,
            models_config=models_config
        )
        
        self.logger.info(f"\nTrained {len(models)} models: {list(models.keys())}")
    
    def evaluate_models(self):
        """
        Step 4: Evaluate all trained models on validation set.
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 4: MODEL EVALUATION")
        self.logger.info("=" * 70)
        
        # Evaluate all models
        results = self.classification_engine.evaluate_all_models(
            self.X_val, self.y_val
        )
        
        return results
    
    def calibrate_and_ensemble(self):
        """
        Step 5: Calibrate probabilities and create ensemble.
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 5: CALIBRATION AND ENSEMBLE")
        self.logger.info("=" * 70)
        
        # Calibrate models
        if self.use_calibration:
            self.classification_engine.calibrate_models(self.X_val, self.y_val)
        
        # Create ensemble
        if self.use_ensemble:
            self.classification_engine.create_ensemble(self.X_val, self.y_val)
    
    def generate_predictions(self, output_filename: str = 'submission.csv'):
        """
        Step 6: Generate predictions for test set and save submission file.
        
        IMPORTANTE: Las predicciones se generan con:
        - Si use_ensemble=True: Usa el ENSEMBLE (fusión de modelos calibrados)
        - Si use_ensemble=False: Usa el MEJOR MODELO INDIVIDUAL (después de calibración si está activada)
        
        El submission.csv usa el modelo FINAL (ensemble o mejor individual según configuración)
        
        Args:
            output_filename: Name of the submission file
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("STEP 6: PREDICTION GENERATION")
        self.logger.info("=" * 70)
        
        # Determinar qué modelo se usa
        if self.use_ensemble and self.classification_engine.final_model is not None:
            model_description = "ENSEMBLE (fusión de modelos calibrados)"
        elif self.use_calibration:
            model_description = f"BEST SINGLE MODEL CALIBRADO ({self.classification_engine.best_model_name})"
        else:
            model_description = f"BEST SINGLE MODEL ({self.classification_engine.best_model_name})"
        
        self.logger.info(f"Generando predicciones con: {model_description}")
        
        # Generate predictions
        self.logger.info("Generating predictions for test set...")
        predictions = self.classification_engine.predict(self.X_test, use_final=True)
        
        # Save submission file
        output_path = self.output_dir / output_filename
        save_submission(predictions, self.test_ids, str(output_path))
        
        self.logger.info(f"Submission file saved to: {output_path}")
        self.logger.info(f"Predictions shape: {predictions.shape}")
        self.logger.info(f"Modelo usado para submission: {model_description}")
    
    def save_models(self):
        """
        Save trained models and summary.
        """
        self.logger.info("\nSaving models and summary...")
        
        ensure_dir(str(self.model_dir))
        
        # Save final model (ensemble or best single)
        final_model_path = self.model_dir / 'final_model.pkl'
        self.classification_engine.save_model(str(final_model_path), model_type='final')
        
        # Save best single model
        best_model_path = self.model_dir / 'best_single_model.pkl'
        self.classification_engine.save_model(str(best_model_path), model_type='best')
        
        # Save summary
        summary_path = self.output_dir / 'training_summary.json'
        self.classification_engine.save_summary(str(summary_path))
        
        self.logger.info(f"Models saved to: {self.model_dir}")
        self.logger.info(f"Summary saved to: {summary_path}")
    
    def run_complete_pipeline(self, save_models: bool = True):
        """
        Execute the complete ML pipeline from start to finish.
        
        Args:
            save_models: Whether to save trained models
        """
        try:
            # Step 1: Load and preprocess
            self.load_and_preprocess_data()
            
            # Step 2: Feature engineering
            self.apply_feature_engineering()
            
            # Step 3: Train models
            self.train_models()
            
            # Step 4: Evaluate models
            self.evaluate_models()
            
            # Step 5: Calibrate and ensemble
            self.calibrate_and_ensemble()
            
            # Step 6: Generate predictions
            self.generate_predictions()
            
            # Save models if requested
            if save_models:
                self.save_models()
            
            self.logger.info("\n" + "=" * 70)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 70)
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
            raise


def main():
    """
    Main entry point with command-line interface.
    """
    parser = argparse.ArgumentParser(
        description='Otto Group Product Classification Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Entrenar solo Random Forest
  python src/main.py --models random_forest
  
  # Entrenar RF y XGBoost sin ensemble
  python src/main.py --models random_forest xgboost --no-ensemble
  
  # Usar modelo guardado para predecir
  python src/main.py --load-model models/final_model.pkl
  
  # Cambiar directorios de salida
  python src/main.py --output-dir custom_outputs --model-dir custom_models
  
  # Configuración completa personalizada
  python src/main.py --models xgboost mlp --no-calibration --pca --output-dir experiment1
        """
    )
    
    # Modo de operación
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--load-model', type=str, metavar='PATH',
                           help='Load pretrained model and predict only (skip training)')
    mode_group.add_argument('--train', action='store_true', default=True,
                           help='Train new models (default mode)')
    
    # Selección de modelos
    parser.add_argument('--models', nargs='+',
                       choices=['random_forest', 'xgboost', 'mlp', 'logistic'],
                       default=['random_forest', 'xgboost', 'mlp', 'logistic'],
                       metavar='MODEL',
                       help='Models to train: random_forest, xgboost, mlp, logistic (default: all)')
    
    # Feature engineering
    parser.add_argument('--pca', action='store_true',
                       help='Use PCA for dimensionality reduction')
    parser.add_argument('--n-components', type=int, default=50,
                       help='Number of PCA components (default: 50)')
    
    # Calibración
    parser.add_argument('--no-calibration', action='store_true',
                       help='Disable probability calibration')
    parser.add_argument('--calibration-method', type=str,
                       choices=['isotonic', 'sigmoid'],
                       default='isotonic',
                       help='Calibration method: isotonic or sigmoid (default: isotonic)')
    
    # Ensemble
    parser.add_argument('--no-ensemble', action='store_true',
                       help='Disable ensemble (use best single model)')
    
    # Directorios
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Directory for output files (default: outputs)')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory for saved models (default: models)')
    
    # Data splitting
    parser.add_argument('--val-size', type=float, default=0.2,
                       help='Validation set size (default: 0.2)')
    
    # Otros
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save models')
    
    args = parser.parse_args()
    
    # Modo 1: Cargar modelo guardado y predecir
    if args.load_model:
        predict_with_loaded_model(
            model_path=args.load_model,
            output_dir=args.output_dir
        )
        return
    
    # Modo 2: Entrenar nuevos modelos
    # Create pipeline
    pipeline = OttoClassificationPipeline(
        use_pca=args.pca,
        n_components=args.n_components,
        use_calibration=not args.no_calibration,
        calibration_method=args.calibration_method,
        use_ensemble=not args.no_ensemble,
        validation_size=args.val_size,
        random_state=args.seed,
        models_to_train=args.models,
        output_dir=args.output_dir,
        model_dir=args.model_dir
    )
    
    # Run complete pipeline
    pipeline.run_complete_pipeline(save_models=not args.no_save)


if __name__ == "__main__":
    main()
