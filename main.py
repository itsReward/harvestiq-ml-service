"""
Maize Yield Prediction ML Model Service
Main FastAPI application for serving machine learning predictions
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date, datetime
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import uvicorn
import os

from models.ensemble_predictor import EnsembleYieldPredictor
from models.feature_engineering import FeatureEngineering
from models.model_training import ModelTrainer
from schemas.prediction_schemas import (
    PredictionRequest,
    PredictionResponse,
    FactorImportance,
    ModelTrainingRequest,
    ModelInfo
)
from utils.data_validation import DataValidator
from utils.model_versioning import ModelVersionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HarvestIQ ML Model Service",
    description="Machine Learning service for maize yield prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model_predictor: Optional[EnsembleYieldPredictor] = None
feature_engineer: Optional[FeatureEngineering] = None
data_validator: Optional[DataValidator] = None
version_manager: Optional[ModelVersionManager] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the ML service on startup"""
    global model_predictor, feature_engineer, data_validator, version_manager

    try:
        logger.info("Starting ML Model Service...")

        # Initialize components
        feature_engineer = FeatureEngineering()
        data_validator = DataValidator()
        version_manager = ModelVersionManager()

        # Load the latest trained model
        model_path = version_manager.get_latest_model_path()
        if model_path and model_path.exists():
            model_predictor = EnsembleYieldPredictor.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            # Initialize with default model if no trained model exists
            model_predictor = EnsembleYieldPredictor()
            logger.warning("No trained model found. Using default model.")

        logger.info("ML Model Service started successfully")

    except Exception as e:
        logger.error(f"Failed to start ML service: {str(e)}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "HarvestIQ ML Model Service",
        "status": "running",
        "version": "1.0.0",
        "model_loaded": model_predictor is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_predictor is not None,
        "feature_engineer_ready": feature_engineer is not None,
        "data_validator_ready": data_validator is not None,
        "version_manager_ready": version_manager is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_yield(request: PredictionRequest):
    """
    Generate yield prediction for a maize planting session
    """
    try:
        logger.info(f"Processing prediction request for farm {request.farm_id}")

        # Validate input data
        if not data_validator.validate_prediction_request(request):
            raise HTTPException(status_code=400, detail="Invalid input data")

        # Check if model is loaded
        if not model_predictor:
            raise HTTPException(status_code=503, detail="Model not available")

        # Engineer features from input data
        features = feature_engineer.create_features(request)

        # Make prediction
        prediction_result = model_predictor.predict(features)

        # Get feature importance for explainability
        feature_importance = model_predictor.get_feature_importance(features)

        # Prepare response
        response = PredictionResponse(
            predicted_yield_tons_per_hectare=round(prediction_result['yield'], 2),
            confidence_percentage=round(prediction_result['confidence'], 2),
            prediction_date=date.today(),
            model_version=model_predictor.get_version(),
            features_used=list(features.keys()),
            important_factors=feature_importance
        )

        logger.info(f"Prediction completed: {response.predicted_yield_tons_per_hectare} tons/ha "
                   f"with {response.confidence_percentage}% confidence")

        return response

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(requests: List[PredictionRequest]):
    """
    Generate predictions for multiple planting sessions
    """
    try:
        results = []
        for request in requests:
            try:
                result = await predict_yield(request)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch prediction failed for farm {request.farm_id}: {str(e)}")
                # Continue with other predictions
                continue

        return {"predictions": results, "total_processed": len(results)}

    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the current model"""
    if not model_predictor:
        raise HTTPException(status_code=503, detail="Model not available")

    return ModelInfo(
        version=model_predictor.get_version(),
        training_date=model_predictor.get_training_date(),
        model_type="Ensemble (Random Forest + LSTM + XGBoost)",
        accuracy_metrics=model_predictor.get_metrics(),
        feature_count=model_predictor.get_feature_count()
    )

@app.post("/model/retrain")
async def retrain_model(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger model retraining with new data
    """
    try:
        # Add retraining task to background
        background_tasks.add_task(
            retrain_model_task,
            request.training_data_path,
            request.model_config
        )

        return {
            "message": "Model retraining initiated",
            "training_id": f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }

    except Exception as e:
        logger.error(f"Failed to initiate retraining: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

async def retrain_model_task(data_path: str, config: Dict[str, Any]):
    """Background task for model retraining"""
    global model_predictor

    try:
        logger.info("Starting model retraining...")

        # Initialize trainer
        trainer = ModelTrainer(config)

        # Train new model
        new_model = trainer.train(data_path)

        # Validate new model performance
        if trainer.validate_model(new_model):
            # Save new model version
            model_path = version_manager.save_new_version(new_model)

            # Update current model
            model_predictor = new_model

            logger.info(f"Model retraining completed. New model saved to {model_path}")
        else:
            logger.warning("New model validation failed. Keeping current model.")

    except Exception as e:
        logger.error(f"Model retraining failed: {str(e)}")

@app.get("/model/versions")
async def get_model_versions():
    """Get list of available model versions"""
    try:
        versions = version_manager.list_versions()
        return {"versions": versions}
    except Exception as e:
        logger.error(f"Failed to get model versions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/load-version/{version}")
async def load_model_version(version: str):
    """Load a specific model version"""
    global model_predictor

    try:
        model_path = version_manager.get_version_path(version)
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model version {version} not found")

        model_predictor = EnsembleYieldPredictor.load_model(model_path)

        return {
            "message": f"Successfully loaded model version {version}",
            "model_info": {
                "version": model_predictor.get_version(),
                "training_date": model_predictor.get_training_date()
            }
        }

    except Exception as e:
        logger.error(f"Failed to load model version {version}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features/importance")
async def get_global_feature_importance():
    """Get global feature importance from the current model"""
    if not model_predictor:
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        importance = model_predictor.get_global_feature_importance()
        return {"feature_importance": importance}
    except Exception as e:
        logger.error(f"Failed to get feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )