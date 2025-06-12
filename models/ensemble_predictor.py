"""
Ensemble Yield Predictor
Combines multiple ML models for accurate maize yield prediction
"""

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import date, datetime
import json

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Feature importance and explainability
import shap

from schemas.prediction_schemas import FactorImportance

logger = logging.getLogger(__name__)

class EnsembleYieldPredictor:
    """
    Ensemble model combining multiple algorithms for maize yield prediction
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.version = f"ensemble_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.training_date = None
        self.feature_names = []
        self.scalers = {}
        self.label_encoders = {}

        # Initialize models
        self.models = {
            'random_forest': RandomForestRegressor(**self.config['random_forest']),
            'xgboost': xgb.XGBRegressor(**self.config['xgboost']),
            'gradient_boosting': GradientBoostingRegressor(**self.config['gradient_boosting'])
        }

        # LSTM model for weather time series
        self.lstm_model = None
        self.weather_scaler = StandardScaler()

        # Meta-learner for ensemble
        self.meta_learner = RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            max_depth=5
        )

        # SHAP explainers for interpretability
        self.explainers = {}

        # Model performance metrics
        self.metrics = {}

        # Training history
        self.training_history = []

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for models"""
        return {
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            },
            'xgboost': {
                'n_estimators': 200,
                'max_depth': 10,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            },
            'gradient_boosting': {
                'n_estimators': 150,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'random_state': 42
            },
            'lstm': {
                'sequence_length': 30,
                'units': 64,
                'dropout': 0.2,
                'epochs': 100,
                'batch_size': 32
            },
            'ensemble': {
                'weights': {
                    'random_forest': 0.3,
                    'xgboost': 0.3,
                    'gradient_boosting': 0.2,
                    'lstm': 0.2
                }
            }
        }

    def prepare_features(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare features from input data
        """
        features = {}

        # Basic farm and planting info
        features['days_since_planting'] = (data['current_date'] - data['planting_date']).days
        features['planting_month'] = data['planting_date'].month
        features['planting_day_of_year'] = data['planting_date'].timetuple().tm_yday

        # Maize variety features
        variety = data.get('maize_variety', {})
        features['variety_maturity_days'] = variety.get('maturity_days', 120)
        features['variety_drought_resistance'] = int(variety.get('drought_resistance', False))
        features['variety_optimal_temp_min'] = variety.get('optimal_temperature_min', 18.0)
        features['variety_optimal_temp_max'] = variety.get('optimal_temperature_max', 30.0)

        # Farm location features
        location = data.get('farm_location', {})
        features['latitude'] = location.get('latitude', 0.0)
        features['longitude'] = location.get('longitude', 0.0)
        features['elevation'] = location.get('elevation', 1000.0)

        # Soil features
        soil = data.get('soil_data')
        if soil:
            features['soil_ph'] = soil.get('ph_level', 6.5)
            features['soil_organic_matter'] = soil.get('organic_matter_percentage', 3.0)
            features['soil_nitrogen'] = soil.get('nitrogen_content', 1.5)
            features['soil_phosphorus'] = soil.get('phosphorus_content', 1.0)
            features['soil_potassium'] = soil.get('potassium_content', 2.0)
            features['soil_moisture'] = soil.get('moisture_content', 25.0)

            # Soil type encoding
            soil_type = soil.get('soil_type', 'loamy')
            if 'soil_type_encoder' in self.label_encoders:
                try:
                    features['soil_type_encoded'] = self.label_encoders['soil_type_encoder'].transform([soil_type])[0]
                except:
                    features['soil_type_encoded'] = 0  # Unknown soil type
            else:
                features['soil_type_encoded'] = 0
        else:
            # Default soil values
            features.update({
                'soil_ph': 6.5, 'soil_organic_matter': 3.0, 'soil_nitrogen': 1.5,
                'soil_phosphorus': 1.0, 'soil_potassium': 2.0, 'soil_moisture': 25.0,
                'soil_type_encoded': 0
            })

        # Weather aggregated features
        weather_data = data.get('weather_data', [])
        if weather_data:
            weather_df = pd.DataFrame([
                {
                    'date': w['date'],
                    'temp_avg': w.get('average_temperature'),
                    'temp_min': w.get('min_temperature'),
                    'temp_max': w.get('max_temperature'),
                    'rainfall': w.get('rainfall_mm', 0),
                    'humidity': w.get('humidity_percentage'),
                    'wind_speed': w.get('wind_speed_kmh')
                }
                for w in weather_data
            ])

            # Calculate weather aggregations
            features['avg_temperature'] = weather_df['temp_avg'].mean() if weather_df['temp_avg'].notna().any() else 25.0
            features['min_temperature'] = weather_df['temp_min'].min() if weather_df['temp_min'].notna().any() else 15.0
            features['max_temperature'] = weather_df['temp_max'].max() if weather_df['temp_max'].notna().any() else 35.0
            features['total_rainfall'] = weather_df['rainfall'].sum()
            features['avg_humidity'] = weather_df['humidity'].mean() if weather_df['humidity'].notna().any() else 70.0
            features['avg_wind_speed'] = weather_df['wind_speed'].mean() if weather_df['wind_speed'].notna().any() else 10.0

            # Temperature stress indicators
            features['heat_stress_days'] = len(weather_df[weather_df['temp_max'] > 35])
            features['cold_stress_days'] = len(weather_df[weather_df['temp_min'] < 10])
            features['temp_variability'] = weather_df['temp_avg'].std() if weather_df['temp_avg'].notna().any() else 5.0

            # Rainfall patterns
            features['dry_spell_length'] = self._calculate_dry_spell(weather_df['rainfall'])
            features['rainfall_variability'] = weather_df['rainfall'].std()

        else:
            # Default weather values
            features.update({
                'avg_temperature': 25.0, 'min_temperature': 15.0, 'max_temperature': 35.0,
                'total_rainfall': 300.0, 'avg_humidity': 70.0, 'avg_wind_speed': 10.0,
                'heat_stress_days': 0, 'cold_stress_days': 0, 'temp_variability': 5.0,
                'dry_spell_length': 0, 'rainfall_variability': 50.0
            })

        # Planting session features
        planting_info = data.get('planting_session', {})
        features['seed_rate'] = planting_info.get('seed_rate_kg_per_hectare', 25.0)
        features['row_spacing'] = planting_info.get('row_spacing_cm', 75.0)
        features['fertilizer_amount'] = planting_info.get('fertilizer_amount_kg_per_hectare', 150.0)

        # Fertilizer type encoding
        fertilizer_type = planting_info.get('fertilizer_type', 'NPK')
        if 'fertilizer_encoder' in self.label_encoders:
            try:
                features['fertilizer_type_encoded'] = self.label_encoders['fertilizer_encoder'].transform([fertilizer_type])[0]
            except:
                features['fertilizer_type_encoded'] = 0
        else:
            features['fertilizer_type_encoded'] = 0

        # Irrigation method encoding
        irrigation = planting_info.get('irrigation_method', 'rainfed')
        if 'irrigation_encoder' in self.label_encoders:
            try:
                features['irrigation_encoded'] = self.label_encoders['irrigation_encoder'].transform([irrigation])[0]
            except:
                features['irrigation_encoded'] = 0
        else:
            features['irrigation_encoded'] = 0

        # Growth stage features
        days_since_planting = features['days_since_planting']
        maturity_days = features['variety_maturity_days']

        features['growth_stage_progress'] = min(days_since_planting / maturity_days, 1.0)
        features['is_vegetative'] = int(days_since_planting < maturity_days * 0.4)
        features['is_reproductive'] = int(maturity_days * 0.4 <= days_since_planting < maturity_days * 0.8)
        features['is_grain_filling'] = int(days_since_planting >= maturity_days * 0.8)

        # Seasonal features
        features['season_sin'] = np.sin(2 * np.pi * features['planting_day_of_year'] / 365)
        features['season_cos'] = np.cos(2 * np.pi * features['planting_day_of_year'] / 365)

        return pd.DataFrame([features])

    def _calculate_dry_spell(self, rainfall_series: pd.Series) -> int:
        """Calculate the longest consecutive dry spell"""
        if rainfall_series.empty:
            return 0

        dry_spell = 0
        max_dry_spell = 0

        for rainfall in rainfall_series:
            if rainfall <= 1.0:  # Consider <= 1mm as dry
                dry_spell += 1
                max_dry_spell = max(max_dry_spell, dry_spell)
            else:
                dry_spell = 0

        return max_dry_spell

    def prepare_weather_sequence(self, weather_data: List[Dict]) -> np.ndarray:
        """
        Prepare weather data for LSTM model
        """
        if not weather_data:
            # Return dummy sequence if no weather data
            sequence_length = self.config['lstm']['sequence_length']
            return np.zeros((1, sequence_length, 6))  # 6 weather features

        # Convert to DataFrame and sort by date
        weather_df = pd.DataFrame(weather_data)
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        weather_df = weather_df.sort_values('date')

        # Select weather features for LSTM
        weather_features = ['average_temperature', 'rainfall_mm', 'humidity_percentage',
                          'wind_speed_kmh', 'min_temperature', 'max_temperature']

        # Fill missing values
        for feature in weather_features:
            if feature not in weather_df.columns:
                weather_df[feature] = 0.0

        weather_df[weather_features] = weather_df[weather_features].fillna(method='forward').fillna(0)

        # Create sequences
        sequence_length = self.config['lstm']['sequence_length']

        if len(weather_df) < sequence_length:
            # Pad with last available values if sequence is too short
            last_values = weather_df[weather_features].iloc[-1:].values
            padding_needed = sequence_length - len(weather_df)
            padding = np.tile(last_values, (padding_needed, 1))
            sequence = np.vstack([padding, weather_df[weather_features].values])
        else:
            # Take the last sequence_length days
            sequence = weather_df[weather_features].iloc[-sequence_length:].values

        # Scale the sequence
        sequence_scaled = self.weather_scaler.transform(sequence.reshape(-1, len(weather_features)))

        return sequence_scaled.reshape(1, sequence_length, len(weather_features))

    def train(self, training_data: pd.DataFrame) -> Dict[str, float]:
        """
        Train the ensemble model
        """
        logger.info("Starting ensemble model training...")

        # Prepare features and target
        X = training_data.drop(['yield'], axis=1)
        y = training_data['yield']

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        self.scalers['feature_scaler'] = StandardScaler()
        X_train_scaled = self.scalers['feature_scaler'].fit_transform(X_train)
        X_test_scaled = self.scalers['feature_scaler'].transform(X_test)

        # Train individual models
        model_predictions_train = {}
        model_predictions_test = {}

        for name, model in self.models.items():
            logger.info(f"Training {name} model...")

            if name == 'xgboost':
                model.fit(X_train, y_train)
                model_predictions_train[name] = model.predict(X_train)
                model_predictions_test[name] = model.predict(X_test)
            else:
                model.fit(X_train_scaled, y_train)
                model_predictions_train[name] = model.predict(X_train_scaled)
                model_predictions_test[name] = model.predict(X_test_scaled)

            # Initialize SHAP explainer
            if name == 'random_forest':
                self.explainers[name] = shap.TreeExplainer(model)
            elif name == 'xgboost':
                self.explainers[name] = shap.TreeExplainer(model)

        # Train LSTM model if weather sequence data is available
        if 'weather_sequences' in training_data.columns:
            logger.info("Training LSTM model...")
            self.lstm_model = self._build_lstm_model()

            # Prepare weather sequences for training
            weather_sequences = np.array(training_data['weather_sequences'].tolist())

            # Train LSTM
            history = self.lstm_model.fit(
                weather_sequences,
                y_train.values,
                epochs=self.config['lstm']['epochs'],
                batch_size=self.config['lstm']['batch_size'],
                validation_split=0.2,
                callbacks=[
                    EarlyStopping(patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=5)
                ],
                verbose=0
            )

            # Get LSTM predictions
            model_predictions_train['lstm'] = self.lstm_model.predict(weather_sequences).flatten()
            model_predictions_test['lstm'] = self.lstm_model.predict(weather_sequences).flatten()

        # Train meta-learner
        logger.info("Training meta-learner...")
        meta_features_train = np.column_stack(list(model_predictions_train.values()))
        meta_features_test = np.column_stack(list(model_predictions_test.values()))

        self.meta_learner.fit(meta_features_train, y_train)

        # Final ensemble predictions
        final_predictions_train = self.meta_learner.predict(meta_features_train)
        final_predictions_test = self.meta_learner.predict(meta_features_test)

        # Calculate metrics
        self.metrics = {
            'train_mae': mean_absolute_error(y_train, final_predictions_train),
            'test_mae': mean_absolute_error(y_test, final_predictions_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, final_predictions_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, final_predictions_test)),
            'train_r2': r2_score(y_train, final_predictions_train),
            'test_r2': r2_score(y_test, final_predictions_test)
        }

        # Calculate MAPE (Mean Absolute Percentage Error)
        self.metrics['train_mape'] = np.mean(np.abs((y_train - final_predictions_train) / y_train)) * 100
        self.metrics['test_mape'] = np.mean(np.abs((y_test - final_predictions_test) / y_test)) * 100

        self.training_date = datetime.now().isoformat()

        logger.info(f"Training completed. Test R²: {self.metrics['test_r2']:.4f}, Test RMSE: {self.metrics['test_rmse']:.4f}")

        return self.metrics

    def _build_lstm_model(self) -> tf.keras.Model:
        """Build LSTM model for weather sequence processing"""
        model = Sequential([
            LSTM(
                self.config['lstm']['units'],
                return_sequences=True,
                input_shape=(self.config['lstm']['sequence_length'], 6)
            ),
            Dropout(self.config['lstm']['dropout']),
            BatchNormalization(),

            LSTM(self.config['lstm']['units'] // 2, return_sequences=False),
            Dropout(self.config['lstm']['dropout']),
            BatchNormalization(),

            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='linear')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Make yield prediction using ensemble
        """
        # Prepare features
        feature_df = self.prepare_features(features)

        # Get predictions from individual models
        model_predictions = {}

        # Scale features for models that need it
        feature_scaled = self.scalers['feature_scaler'].transform(feature_df)

        for name, model in self.models.items():
            if name == 'xgboost':
                model_predictions[name] = model.predict(feature_df)[0]
            else:
                model_predictions[name] = model.predict(feature_scaled)[0]

        # LSTM prediction if available
        if self.lstm_model and 'weather_data' in features:
            weather_sequence = self.prepare_weather_sequence(features['weather_data'])
            model_predictions['lstm'] = self.lstm_model.predict(weather_sequence)[0, 0]

        # Meta-learner prediction
        meta_features = np.array(list(model_predictions.values())).reshape(1, -1)
        final_prediction = self.meta_learner.predict(meta_features)[0]

        # Calculate confidence based on model agreement
        pred_values = list(model_predictions.values())
        confidence = self._calculate_confidence(pred_values, final_prediction)

        return {
            'yield': max(0.0, final_prediction),  # Ensure non-negative yield
            'confidence': confidence,
            'individual_predictions': model_predictions
        }

    def _calculate_confidence(self, predictions: List[float], final_prediction: float) -> float:
        """Calculate prediction confidence based on model agreement"""
        if len(predictions) < 2:
            return 75.0  # Default confidence

        # Calculate coefficient of variation
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)

        if mean_pred == 0:
            return 50.0

        cv = std_pred / abs(mean_pred)

        # Convert to confidence (lower variation = higher confidence)
        confidence = max(50.0, min(95.0, 95.0 - cv * 100))

        return confidence

    def get_feature_importance(self, features: Dict[str, Any]) -> List[FactorImportance]:
        """
        Get feature importance for explainability using SHAP
        """
        try:
            feature_df = self.prepare_features(features)
            feature_scaled = self.scalers['feature_scaler'].transform(feature_df)

            # Get SHAP values from Random Forest (most interpretable)
            explainer = self.explainers.get('random_forest')
            if explainer:
                shap_values = explainer.shap_values(feature_scaled)

                # Create factor importance list
                importance_list = []
                for i, feature_name in enumerate(self.feature_names):
                    importance_value = abs(shap_values[0][i])
                    impact = "POSITIVE" if shap_values[0][i] > 0 else "NEGATIVE"

                    # Map technical feature names to user-friendly names
                    friendly_name = self._get_friendly_feature_name(feature_name)

                    importance_list.append(FactorImportance(
                        factor=friendly_name,
                        importance=float(importance_value),
                        impact=impact,
                        description=self._get_feature_description(feature_name)
                    ))

                # Sort by importance and return top 10
                importance_list.sort(key=lambda x: x.importance, reverse=True)
                return importance_list[:10]

        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")

        # Fallback: return default important factors
        return self._get_default_important_factors()

    def _get_friendly_feature_name(self, feature_name: str) -> str:
        """Convert technical feature names to user-friendly names"""
        name_mapping = {
            'days_since_planting': 'Days Since Planting',
            'avg_temperature': 'Average Temperature',
            'total_rainfall': 'Total Rainfall',
            'soil_ph': 'Soil pH Level',
            'soil_nitrogen': 'Soil Nitrogen Content',
            'soil_organic_matter': 'Soil Organic Matter',
            'variety_drought_resistance': 'Drought Resistance',
            'heat_stress_days': 'Heat Stress Days',
            'fertilizer_amount': 'Fertilizer Application',
            'growth_stage_progress': 'Growth Stage'
        }
        return name_mapping.get(feature_name, feature_name.replace('_', ' ').title())

    def _get_feature_description(self, feature_name: str) -> str:
        """Get description for feature"""
        descriptions = {
            'days_since_planting': 'Number of days elapsed since planting',
            'avg_temperature': 'Average temperature during growing period',
            'total_rainfall': 'Cumulative rainfall received',
            'soil_ph': 'Soil acidity/alkalinity level',
            'soil_nitrogen': 'Available nitrogen in soil',
            'heat_stress_days': 'Days with excessive heat stress'
        }
        return descriptions.get(feature_name, '')

    def _get_default_important_factors(self) -> List[FactorImportance]:
        """Return default important factors when SHAP fails"""
        return [
            FactorImportance(factor="Weather Conditions", importance=0.35, impact="POSITIVE"),
            FactorImportance(factor="Soil Quality", importance=0.25, impact="POSITIVE"),
            FactorImportance(factor="Growth Stage", importance=0.20, impact="NEUTRAL"),
            FactorImportance(factor="Maize Variety", importance=0.15, impact="POSITIVE"),
            FactorImportance(factor="Management Practices", importance=0.05, impact="POSITIVE")
        ]

    def get_global_feature_importance(self) -> List[FactorImportance]:
        """Get global feature importance from trained models"""
        if not self.models or 'random_forest' not in self.models:
            return self._get_default_important_factors()

        try:
            # Get feature importance from Random Forest
            rf_model = self.models['random_forest']
            feature_importance = rf_model.feature_importances_

            importance_list = []
            for i, importance in enumerate(feature_importance):
                if i < len(self.feature_names):
                    feature_name = self.feature_names[i]
                    friendly_name = self._get_friendly_feature_name(feature_name)

                    importance_list.append(FactorImportance(
                        factor=friendly_name,
                        importance=float(importance),
                        impact="POSITIVE",  # Global importance doesn't indicate direction
                        description=self._get_feature_description(feature_name)
                    ))

            # Sort by importance and return top 10
            importance_list.sort(key=lambda x: x.importance, reverse=True)
            return importance_list[:10]

        except Exception as e:
            logger.error(f"Error getting global feature importance: {str(e)}")
            return self._get_default_important_factors()

    def get_version(self) -> str:
        """Get model version"""
        return self.version

    def get_training_date(self) -> Optional[str]:
        """Get training date"""
        return self.training_date

    def get_metrics(self) -> Dict[str, float]:
        """Get model performance metrics"""
        return self.metrics

    def get_feature_count(self) -> int:
        """Get number of features"""
        return len(self.feature_names)

    def save_model(self, save_path: Path) -> None:
        """Save the trained model"""
        save_path.mkdir(parents=True, exist_ok=True)

        # Save sklearn models
        joblib.dump(self.models, save_path / 'sklearn_models.joblib')
        joblib.dump(self.meta_learner, save_path / 'meta_learner.joblib')
        joblib.dump(self.scalers, save_path / 'scalers.joblib')
        joblib.dump(self.label_encoders, save_path / 'label_encoders.joblib')

        # Save LSTM model if available
        if self.lstm_model:
            self.lstm_model.save(save_path / 'lstm_model.h5')

        # Save metadata
        metadata = {
            'version': self.version,
            'training_date': self.training_date,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'config': self.config
        }

        with open(save_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {save_path}")

    @classmethod
    def load_model(cls, load_path: Path) -> 'EnsembleYieldPredictor':
        """Load a saved model"""
        # Load metadata
        with open(load_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        # Create instance
        instance = cls(metadata['config'])
        instance.version = metadata['version']
        instance.training_date = metadata['training_date']
        instance.feature_names = metadata['feature_names']
        instance.metrics = metadata['metrics']

        # Load sklearn models
        instance.models = joblib.load(load_path / 'sklearn_models.joblib')
        instance.meta_learner = joblib.load(load_path / 'meta_learner.joblib')
        instance.scalers = joblib.load(load_path / 'scalers.joblib')
        instance.label_encoders = joblib.load(load_path / 'label_encoders.joblib')

        # Load LSTM model if available
        lstm_path = load_path / 'lstm_model.h5'
        if lstm_path.exists():
            instance.lstm_model = tf.keras.models.load_model(lstm_path)

        # Initialize SHAP explainers
        for name, model in instance.models.items():
            if name in ['random_forest', 'xgboost']:
                instance.explainers[name] = shap.TreeExplainer(model)

        logger.info(f"Model loaded from {load_path}")
        return instance