#!/usr/bin/env python3
"""
Model Training Script for Maize Yield Prediction
Handles training, validation, and hyperparameter optimization
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import project modules
from models.ensemble_predictor import EnsembleYieldPredictor
from models.feature_engineering import FeatureEngineering
from models.model_training import ModelTrainer
from utils.data_validation import DataValidator
from utils.model_versioning import ModelVersionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def suppress_tensorflow_warnings():
    """Suppress TensorFlow warnings"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

def generate_synthetic_data(num_samples: int = 1000) -> List[Dict]:
    """Generate synthetic training data for testing"""
    logger.info(f"Generating {num_samples} synthetic training samples...")

    import random
    from datetime import date, timedelta

    synthetic_data = []
    base_date = date(2020, 1, 1)

    for i in range(num_samples):
        # Generate basic farm info
        farm_id = random.randint(1, 100)

        # Generate planting date
        planting_date = base_date + timedelta(days=random.randint(0, 1460))  # 4 years range
        harvest_date = planting_date + timedelta(days=random.randint(110, 130))

        # Generate variety info
        varieties = ['ZM 523', 'ZM 621', 'PAN 413', 'SC 403', 'ZM 309']
        variety_name = random.choice(varieties)
        maturity_days = random.randint(110, 130)
        drought_resistance = random.choice([True, False])

        # Generate soil data
        soil_types = ['Sandy loam', 'Clay loam', 'Loam', 'Sandy clay', 'Silt loam']
        soil_type = random.choice(soil_types)
        ph_level = random.uniform(5.5, 7.5)
        organic_matter = random.uniform(1.0, 4.0)
        nitrogen = random.uniform(0.1, 0.3)
        phosphorus = random.uniform(10, 50)
        potassium = random.uniform(100, 300)

        # Generate weather data (simplified)
        avg_temp = random.uniform(20, 30)
        total_rainfall = random.uniform(300, 800)
        humidity = random.uniform(60, 85)

        # Calculate yield based on factors (simplified model)
        base_yield = 4.0  # Base yield in tons per hectare

        # Variety factor
        variety_factor = 1.1 if drought_resistance else 1.0

        # Soil factor
        soil_factor = 1.0
        if 6.0 <= ph_level <= 7.0:
            soil_factor += 0.1
        if organic_matter > 2.5:
            soil_factor += 0.1

        # Weather factor
        weather_factor = 1.0
        if 400 <= total_rainfall <= 600:
            weather_factor += 0.2
        if 22 <= avg_temp <= 28:
            weather_factor += 0.1

        # Calculate final yield with some noise
        yield_tons = base_yield * variety_factor * soil_factor * weather_factor
        yield_tons += random.normal(0, 0.3)  # Add noise
        yield_tons = max(0.5, min(8.0, yield_tons))  # Clamp between reasonable bounds

        # Create synthetic record
        record = {
            'farm_id': farm_id,
            'planting_session_id': i + 1,
            'harvest_date': harvest_date.isoformat(),
            'actual_yield_tons_per_hectare': round(yield_tons, 2),
            'variety_id': random.randint(1, 5),
            'variety_name': variety_name,
            'maturity_days': maturity_days,
            'drought_resistance': drought_resistance,
            'planting_date': planting_date.isoformat(),
            'field_size_hectares': random.uniform(0.5, 5.0),
            'plant_density_per_hectare': random.randint(25000, 35000),
            'fertilizer_applied': random.choice([True, False]),
            'irrigation_used': random.choice([True, False]),
            'soil_type': soil_type,
            'ph_level': round(ph_level, 1),
            'organic_matter_percentage': round(organic_matter, 1),
            'nitrogen_content': round(nitrogen, 2),
            'phosphorus_content': round(phosphorus, 1),
            'potassium_content': round(potassium, 1),
            'avg_temperature': round(avg_temp, 1),
            'total_rainfall': round(total_rainfall, 1),
            'avg_humidity': round(humidity, 1),
            'latitude': random.uniform(-22.0, -15.0),  # Zimbabwe coordinates
            'longitude': random.uniform(25.0, 33.0),
            'elevation': random.uniform(500, 1500)
        }

        synthetic_data.append(record)

    logger.info(f"Generated {len(synthetic_data)} synthetic records")
    return synthetic_data

def load_training_data(data_path: str, create_synthetic: bool = False) -> List[Dict]:
    """Load training data from file or generate synthetic data"""
    data_path = Path(data_path)

    if create_synthetic or not data_path.exists():
        if not data_path.exists():
            logger.warning(f"Data file {data_path} not found. Generating synthetic data.")
        return generate_synthetic_data(2000)

    logger.info(f"Loading training data from {data_path}")

    if data_path.suffix.lower() == '.json':
        with open(data_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} records from JSON file")
        return data
    elif data_path.suffix.lower() == '.csv':
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} records from CSV file")
        return df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

def save_training_data(data: List[Dict], output_path: str):
    """Save training data to file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(data)} records to {output_path}")

def train_model(data_path: str, create_synthetic: bool = False,
               output_dir: str = None, config: Dict = None) -> EnsembleYieldPredictor:
    """Main training function"""

    # Load data
    training_data = load_training_data(data_path, create_synthetic)

    if create_synthetic:
        # Save synthetic data for future use
        synthetic_path = Path(data_path).parent / f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_training_data(training_data, str(synthetic_path))

    # Initialize components
    logger.info("Initializing training components...")
    trainer = ModelTrainer(config)

    # Train model
    logger.info("Starting model training...")
    model = trainer.train_from_data(training_data)

    # Save model
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")

    return model

def main():
    """Main training script"""
    suppress_tensorflow_warnings()

    parser = argparse.ArgumentParser(description='Train maize yield prediction model')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to training data file')
    parser.add_argument('--create-synthetic', action='store_true',
                        help='Generate synthetic data if data file not found')
    parser.add_argument('--output-dir', type=str, default='./storage/models',
                        help='Directory to save trained model')
    parser.add_argument('--config', type=str,
                        help='Path to training configuration JSON file')
    parser.add_argument('--quick-train', action='store_true',
                        help='Quick training with reduced hyperparameter search')

    args = parser.parse_args()

    # Load configuration
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    elif args.quick_train:
        # Quick training configuration
        config = {
            'hyperparameter_optimization': {
                'enabled': False
            },
            'validation_split': 0.2,
            'test_split': 0.1
        }

    try:
        # Train model
        logger.info("🌾 Starting HarvestIQ ML Model Training...")
        logger.info(f"📊 Data path: {args.data_path}")
        logger.info(f"💾 Output directory: {args.output_dir}")
        logger.info(f"🔧 Create synthetic: {args.create_synthetic}")

        model = train_model(
            data_path=args.data_path,
            create_synthetic=args.create_synthetic,
            output_dir=args.output_dir,
            config=config
        )

        logger.info("✅ Model training completed successfully!")
        logger.info(f"📈 Model version: {model.version}")

        # Print basic model info
        print(f"\n🎯 Training Results:")
        print(f"   Model Version: {model.version}")
        print(f"   Feature Count: {len(model.feature_names) if model.feature_names else 'Unknown'}")
        if hasattr(model, 'metrics') and model.metrics:
            for metric, value in model.metrics.items():
                print(f"   {metric.upper()}: {value:.4f}")

        return 0

    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())