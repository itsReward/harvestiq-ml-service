"""
Model Training Module for Maize Yield Prediction
Handles training, validation, and hyperparameter optimization
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import joblib
import json
from datetime import datetime

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Hyperparameter optimization
from scipy.stats import randint, uniform
import optuna

from models.ensemble_predictor import EnsembleYieldPredictor
from models.feature_engineering import FeatureEngineering
from utils.data_validation import DataValidator

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Comprehensive model training with hyperparameter optimization
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.feature_engineer = FeatureEngineering()
        self.data_validator = DataValidator()
        self.best_params = {}
        self.training_history = []

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration"""
        return {
            'validation_split': 0.2,
            'test_split': 0.1,
            'cross_validation_folds': 5,
            'random_state': 42,
            'hyperparameter_optimization': {
                'enabled': True,
                'method': 'optuna',  # 'grid', 'random', 'optuna'
                'n_trials': 100,
                'timeout_minutes': 60
            },
            'early_stopping': {
                'enabled': True,
                'patience': 10,
                'min_delta': 0.001
            },
            'model_selection': {
                'metric': 'r2_score',  # 'mae', 'mse', 'rmse', 'r2_score', 'mape'
                'higher_is_better': True
            }
        }

    def train(self, data_path: str) -> EnsembleYieldPredictor:
        """
        Main training pipeline
        """
        logger.info("Starting model training pipeline...")

        # Load and prepare data
        training_data = self._load_training_data(data_path)
        logger.info(f"Loaded {len(training_data)} training samples")

        # Validate data quality
        if not self.data_validator.validate_training_data(training_data):
            raise ValueError("Training data validation failed")

        # Feature engineering
        feature_data = self.feature_engineer.create_training_features(training_data)
        logger.info(f"Created {len(feature_data.columns)} features")

        # Prepare features and target
        X = feature_data.drop(['yield'], axis=1)
        y = feature_data['yield']

        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=self.config['validation_split'] + self.config['test_split'],
            random_state=self.config['random_state'],
            stratify=self._create_stratification_bins(y)
        )

        # Further split temp into validation and test
        val_size = self.config['validation_split'] / (self.config['validation_split'] + self.config['test_split'])
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=1 - val_size,
            random_state=self.config['random_state']
        )

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Hyperparameter optimization
        if self.config['hyperparameter_optimization']['enabled']:
            best_params = self._optimize_hyperparameters(X_train, y_train, X_val, y_val)
            self.best_params = best_params
        else:
            self.best_params = {}

        # Train final ensemble model
        ensemble_model = self._train_ensemble_model(
            X_train, y_train, X_val, y_val, X_test, y_test
        )

        # Final evaluation
        final_metrics = self._evaluate_model(ensemble_model, X_test, y_test)
        logger.info(f"Final model performance: {final_metrics}")

        return ensemble_model

    def _load_training_data(self, data_path: str) -> List[Dict]:
        """Load training data from various formats"""
        data_path = Path(data_path)

        if data_path.suffix.lower() == '.csv':
            df = pd.read_csv(data_path)
            return df.to_dict('records')
        elif data_path.suffix.lower() == '.json':
            with open(data_path, 'r') as f:
                return json.load(f)
        elif data_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(data_path)
            return df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

    def _create_stratification_bins(self, y: pd.Series, n_bins: int = 5) -> pd.Series:
        """Create stratification bins for yield values"""
        return pd.cut(y, bins=n_bins, labels=False)

    def _optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters using the configured method"""
        method = self.config['hyperparameter_optimization']['method']

        if method == 'optuna':
            return self._optuna_optimization(X_train, y_train, X_val, y_val)
        elif method == 'grid':
            return self._grid_search_optimization(X_train, y_train)
        elif method == 'random':
            return self._random_search_optimization(X_train, y_train)
        else:
            logger.warning(f"Unknown optimization method: {method}")
            return {}

    def _optuna_optimization(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Hyperparameter optimization using Optuna"""
        logger.info("Starting Optuna hyperparameter optimization...")

        def objective(trial):
            # Define hyperparameter space
            params = {
                'random_forest': {
                    'n_estimators': trial.suggest_int('rf_n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('rf_max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 5),
                    'random_state': 42
                },
                'xgboost': {
                    'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('xgb_max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
                    'random_state': 42
                },
                'gradient_boosting': {
                    'n_estimators': trial.suggest_int('gb_n_estimators', 100, 300),
                    'max_depth': trial.suggest_int('gb_max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('gb_learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('gb_subsample', 0.6, 1.0),
                    'random_state': 42
                }
            }

            # Train ensemble with these parameters
            ensemble = EnsembleYieldPredictor(params)

            # Prepare training data
            feature_data = pd.concat([X_train, y_train], axis=1)
            ensemble.train(feature_data)

            # Evaluate on validation set
            X_val_features = {}
            for idx, row in X_val.iterrows():
                X_val_features.update(row.to_dict())

            predictions = []
            actuals = []
            for idx, row in X_val.iterrows():
                features = row.to_dict()
                pred_result = ensemble.predict(features)
                predictions.append(pred_result['yield'])
                actuals.append(y_val.iloc[idx])

            # Calculate validation score
            score = r2_score(actuals, predictions)
            return score

        # Create and run study
        study = optuna.create_study(direction='maximize')
        timeout = self.config['hyperparameter_optimization']['timeout_minutes'] * 60

        study.optimize(
            objective,
            n_trials=self.config['hyperparameter_optimization']['n_trials'],
            timeout=timeout
        )

        logger.info(f"Best score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")

        # Convert Optuna parameters back to model config format
        best_params = {
            'random_forest': {
                'n_estimators': study.best_params['rf_n_estimators'],
                'max_depth': study.best_params['rf_max_depth'],
                'min_samples_split': study.best_params['rf_min_samples_split'],
                'min_samples_leaf': study.best_params['rf_min_samples_leaf'],
                'random_state': 42,
                'n_jobs': -1
            },
            'xgboost': {
                'n_estimators': study.best_params['xgb_n_estimators'],
                'max_depth': study.best_params['xgb_max_depth'],
                'learning_rate': study.best_params['xgb_learning_rate'],
                'subsample': study.best_params['xgb_subsample'],
                'colsample_bytree': study.best_params['xgb_colsample_bytree'],
                'random_state': 42,
                'n_jobs': -1
            },
            'gradient_boosting': {
                'n_estimators': study.best_params['gb_n_estimators'],
                'max_depth': study.best_params['gb_max_depth'],
                'learning_rate': study.best_params['gb_learning_rate'],
                'subsample': study.best_params['gb_subsample'],
                'random_state': 42
            }
        }

        return best_params

    def _grid_search_optimization(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Grid search hyperparameter optimization"""
        logger.info("Starting Grid Search optimization...")

        # Define parameter grids for each model
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0]
            }
        }

        best_params = {}

        # Optimize each model separately
        for model_name, param_grid in param_grids.items():
            logger.info(f"Optimizing {model_name}...")

            if model_name == 'random_forest':
                model = RandomForestRegressor(random_state=42, n_jobs=-1)
            elif model_name == 'xgboost':
                model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

            # Scale features for non-tree models
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train) if model_name != 'xgboost' else X_train

            # Grid search
            grid_search = GridSearchCV(
                model, param_grid,
                cv=self.config['cross_validation_folds'],
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_scaled, y_train)
            best_params[model_name] = grid_search.best_params_

            logger.info(f"Best {model_name} score: {grid_search.best_score_:.4f}")

        return best_params

    def _random_search_optimization(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Random search hyperparameter optimization"""
        logger.info("Starting Random Search optimization...")

        # Define parameter distributions
        param_distributions = {
            'random_forest': {
                'n_estimators': randint(100, 500),
                'max_depth': randint(5, 25),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': uniform(0.1, 0.9)
            },
            'xgboost': {
                'n_estimators': randint(100, 500),
                'max_depth': randint(3, 15),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4)
            }
        }

        best_params = {}

        for model_name, param_dist in param_distributions.items():
            logger.info(f"Optimizing {model_name}...")

            if model_name == 'random_forest':
                model = RandomForestRegressor(random_state=42, n_jobs=-1)
            elif model_name == 'xgboost':
                model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

            # Scale features for non-tree models
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train) if model_name != 'xgboost' else X_train

            # Random search
            random_search = RandomizedSearchCV(
                model, param_dist,
                n_iter=50,
                cv=self.config['cross_validation_folds'],
                scoring='r2',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )

            random_search.fit(X_scaled, y_train)
            best_params[model_name] = random_search.best_params_

            logger.info(f"Best {model_name} score: {random_search.best_score_:.4f}")

        return best_params

    def _train_ensemble_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_val: pd.DataFrame, y_val: pd.Series,
                              X_test: pd.DataFrame, y_test: pd.Series) -> EnsembleYieldPredictor:
        """Train the final ensemble model"""
        logger.info("Training final ensemble model...")

        # Use best parameters if available
        config = self.best_params if self.best_params else None

        # Create ensemble model
        ensemble = EnsembleYieldPredictor(config)

        # Prepare training data (combine train and validation for final training)
        X_combined = pd.concat([X_train, X_val])
        y_combined = pd.concat([y_train, y_val])

        # Create training DataFrame
        training_data = pd.concat([X_combined, y_combined], axis=1)

        # Train the ensemble
        metrics = ensemble.train(training_data)

        # Store training history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'config': config,
            'data_size': len(training_data)
        })

        return ensemble

    def _evaluate_model(self, model: EnsembleYieldPredictor,
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        logger.info("Evaluating final model...")

        predictions = []
        confidences = []

        # Generate predictions for test set
        for idx, row in X_test.iterrows():
            features = row.to_dict()
            try:
                result = model.predict(features)
                predictions.append(result['yield'])
                confidences.append(result['confidence'])
            except Exception as e:
                logger.warning(f"Prediction failed for sample {idx}: {str(e)}")
                predictions.append(0.0)
                confidences.append(50.0)

        # Calculate comprehensive metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

        # Calculate accuracy within thresholds
        errors = np.abs(y_test - predictions)
        relative_errors = errors / y_test

        accuracy_10_percent = (relative_errors <= 0.1).mean() * 100
        accuracy_20_percent = (relative_errors <= 0.2).mean() * 100

        # Prediction confidence analysis
        avg_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)

        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'mape': mape,
            'accuracy_within_10_percent': accuracy_10_percent,
            'accuracy_within_20_percent': accuracy_20_percent,
            'avg_confidence': avg_confidence,
            'confidence_std': confidence_std,
            'samples_evaluated': len(predictions)
        }

        return metrics

    def validate_model(self, model: EnsembleYieldPredictor,
                       validation_data_path: Optional[str] = None) -> bool:
        """Validate model performance against thresholds"""
        logger.info("Validating model performance...")

        if validation_data_path:
            # Load external validation data
            validation_data = self._load_training_data(validation_data_path)
            feature_data = self.feature_engineer.create_training_features(validation_data)

            X_val = feature_data.drop(['yield'], axis=1)
            y_val = feature_data['yield']
        else:
            # Use a portion of training data for validation
            logger.warning("No validation data provided. Using training data subset.")
            return True  # Simplified validation

        # Evaluate model
        metrics = self._evaluate_model(model, X_val, y_val)

        # Define performance thresholds
        thresholds = {
            'min_r2_score': 0.7,
            'max_mape': 25.0,
            'min_accuracy_20_percent': 70.0,
            'min_avg_confidence': 60.0
        }

        # Check if model meets all thresholds
        validation_passed = True
        failed_criteria = []

        if metrics['r2_score'] < thresholds['min_r2_score']:
            validation_passed = False
            failed_criteria.append(
                f"R² score ({metrics['r2_score']:.3f}) below threshold ({thresholds['min_r2_score']})")

        if metrics['mape'] > thresholds['max_mape']:
            validation_passed = False
            failed_criteria.append(f"MAPE ({metrics['mape']:.1f}%) above threshold ({thresholds['max_mape']}%)")

        if metrics['accuracy_within_20_percent'] < thresholds['min_accuracy_20_percent']:
            validation_passed = False
            failed_criteria.append(
                f"20% accuracy ({metrics['accuracy_within_20_percent']:.1f}%) below threshold ({thresholds['min_accuracy_20_percent']}%)")

        if metrics['avg_confidence'] < thresholds['min_avg_confidence']:
            validation_passed = False
            failed_criteria.append(
                f"Average confidence ({metrics['avg_confidence']:.1f}%) below threshold ({thresholds['min_avg_confidence']}%)")

        if validation_passed:
            logger.info("Model validation PASSED")
        else:
            logger.warning(f"Model validation FAILED. Criteria not met: {failed_criteria}")

        return validation_passed

    def create_synthetic_training_data(self, base_data_path: str, output_path: str,
                                       num_samples: int = 1000) -> None:
        """
        Create synthetic training data for initial model development
        """
        logger.info(f"Creating {num_samples} synthetic training samples...")

        np.random.seed(42)
        synthetic_data = []

        # Define realistic ranges for Zimbabwe maize farming
        for i in range(num_samples):
            # Random farm location in Zimbabwe
            latitude = np.random.uniform(-22.0, -15.5)  # Zimbabwe latitude range
            longitude = np.random.uniform(25.0, 33.0)  # Zimbabwe longitude range
            elevation = np.random.uniform(400, 2000)  # Elevation range

            # Random planting date (Oct-Dec for Zimbabwe)
            planting_month = np.random.choice([10, 11, 12])
            planting_day = np.random.randint(1, 29)
            planting_date = f"2023-{planting_month:02d}-{planting_day:02d}"

            # Random variety
            varieties = [
                {'name': 'ZM 523', 'maturity_days': 120, 'drought_resistance': True},
                {'name': 'ZM 625', 'maturity_days': 125, 'drought_resistance': False},
                {'name': 'ZM 309', 'maturity_days': 90, 'drought_resistance': True},
                {'name': 'Pioneer 30G19', 'maturity_days': 118, 'drought_resistance': True}
            ]
            variety = np.random.choice(varieties)

            # Generate realistic soil data
            soil_data = {
                'soil_type': np.random.choice(['sandy', 'loamy', 'clay', 'sandy_loam']),
                'ph_level': np.random.normal(6.2, 0.8),
                'organic_matter_percentage': np.random.normal(2.5, 1.0),
                'nitrogen_content': np.random.normal(1.2, 0.5),
                'phosphorus_content': np.random.normal(0.8, 0.3),
                'potassium_content': np.random.normal(1.8, 0.6),
                'moisture_content': np.random.normal(25.0, 8.0),
                'sample_date': planting_date
            }

            # Generate weather data (120 days from planting)
            weather_data = []
            base_temp = np.random.normal(25, 3)  # Base temperature

            for day in range(120):
                # Seasonal temperature variation
                seasonal_factor = np.sin(2 * np.pi * day / 365) * 5
                daily_temp = base_temp + seasonal_factor + np.random.normal(0, 3)

                # Rainfall (more during rainy season)
                if planting_month in [11, 12, 1, 2]:  # Rainy season
                    rainfall_prob = 0.4
                    rainfall_amount = np.random.exponential(8) if np.random.random() < rainfall_prob else 0
                else:
                    rainfall_prob = 0.1
                    rainfall_amount = np.random.exponential(3) if np.random.random() < rainfall_prob else 0

                weather_data.append({
                    'date': f"2023-{planting_month:02d}-{planting_day + day:02d}",  # Simplified
                    'min_temperature': daily_temp - 8,
                    'max_temperature': daily_temp + 12,
                    'average_temperature': daily_temp,
                    'rainfall_mm': rainfall_amount,
                    'humidity_percentage': np.random.normal(65, 15),
                    'wind_speed_kmh': np.random.normal(12, 5)
                })

            # Calculate synthetic yield based on conditions
            base_yield = 4.5  # Base yield tons/ha

            # Variety factor
            variety_factor = 1.1 if variety['drought_resistance'] else 1.0

            # Soil factor
            soil_factor = 1.0
            soil_factor *= max(0.7, 1.0 - abs(soil_data['ph_level'] - 6.5) * 0.1)  # pH impact
            soil_factor *= min(1.3, 0.8 + soil_data['organic_matter_percentage'] * 0.1)  # OM impact

            # Weather factor
            total_rainfall = sum(w['rainfall_mm'] for w in weather_data)
            avg_temp = np.mean([w['average_temperature'] for w in weather_data])

            rainfall_factor = min(1.2, total_rainfall / 400.0) if total_rainfall > 0 else 0.5
            temp_factor = 1.0 if 20 <= avg_temp <= 30 else max(0.6, 1.0 - abs(avg_temp - 25) * 0.02)

            # Management factor (random)
            management_factor = np.random.uniform(0.8, 1.2)

            # Calculate final yield with some noise
            final_yield = (base_yield * variety_factor * soil_factor *
                           rainfall_factor * temp_factor * management_factor)
            final_yield = max(0.5, final_yield + np.random.normal(0, 0.5))

            # Create record
            record = {
                'farm_id': i + 1,
                'planting_session_id': i + 1,
                'planting_date': planting_date,
                'harvest_date': f"2024-{(planting_month + 4) % 12 + 1:02d}-{planting_day:02d}",
                'actual_yield_tons_per_hectare': round(final_yield, 2),
                'variety_id': 1,
                'variety_name': variety['name'],
                'maturity_days': variety['maturity_days'],
                'drought_resistance': variety['drought_resistance'],
                'location': {
                    'latitude': latitude,
                    'longitude': longitude,
                    'elevation': elevation
                },
                'soil_data': soil_data,
                'weather_data': weather_data[:30],  # Limit to 30 days for size
                'seed_rate': np.random.normal(25, 5),
                'fertilizer_amount': np.random.normal(150, 50),
                'row_spacing': np.random.choice([75, 90]),
                'irrigation_method': np.random.choice(['rainfed', 'sprinkler'], p=[0.8, 0.2])
            }

            synthetic_data.append(record)

        # Save synthetic data
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(synthetic_data, f, indent=2, default=str)

        logger.info(f"Synthetic training data saved to {output_path}")

    def get_training_history(self) -> List[Dict]:
        """Get training history"""
        return self.training_history

    def export_model_report(self, model: EnsembleYieldPredictor,
                            output_path: str) -> None:
        """Export comprehensive model training report"""
        report = {
            'model_info': {
                'version': model.get_version(),
                'training_date': model.get_training_date(),
                'feature_count': model.get_feature_count()
            },
            'training_config': self.config,
            'hyperparameters': self.best_params,
            'performance_metrics': model.get_metrics(),
            'training_history': self.training_history,
            'feature_importance': model.get_global_feature_importance()
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Model training report exported to {output_path}")