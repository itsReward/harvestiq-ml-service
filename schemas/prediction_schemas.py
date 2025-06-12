"""
Pydantic schemas for the ML prediction service
Matches the DTOs from the Kotlin backend
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import date
from decimal import Decimal

class FarmLocationData(BaseModel):
    """Farm location information"""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    elevation: Optional[float] = None

class SoilDataRequest(BaseModel):
    """Soil data for prediction"""
    soil_type: str
    ph_level: Optional[float] = None
    organic_matter_percentage: Optional[float] = None
    nitrogen_content: Optional[float] = None
    phosphorus_content: Optional[float] = None
    potassium_content: Optional[float] = None
    moisture_content: Optional[float] = None
    sample_date: date

    @validator('ph_level')
    def validate_ph(cls, v):
        if v is not None and (v < 0 or v > 14):
            raise ValueError('pH level must be between 0 and 14')
        return v

    @validator('organic_matter_percentage', 'nitrogen_content', 'phosphorus_content', 'potassium_content', 'moisture_content')
    def validate_percentages(cls, v):
        if v is not None and v < 0:
            raise ValueError('Content percentages must be non-negative')
        return v

class WeatherDataRequest(BaseModel):
    """Weather data for prediction"""
    date: date
    min_temperature: Optional[float] = None
    max_temperature: Optional[float] = None
    average_temperature: Optional[float] = None
    rainfall_mm: Optional[float] = None
    humidity_percentage: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    solar_radiation: Optional[float] = None

    @validator('humidity_percentage')
    def validate_humidity(cls, v):
        if v is not None and (v < 0 or v > 100):
            raise ValueError('Humidity must be between 0 and 100')
        return v

    @validator('rainfall_mm', 'wind_speed_kmh', 'solar_radiation')
    def validate_non_negative(cls, v):
        if v is not None and v < 0:
            raise ValueError('Value must be non-negative')
        return v

class MaizeVarietyInfo(BaseModel):
    """Maize variety characteristics"""
    id: int
    name: str
    maturity_days: int
    optimal_temperature_min: Optional[float] = None
    optimal_temperature_max: Optional[float] = None
    drought_resistance: bool = False
    disease_resistance: Optional[str] = None
    description: Optional[str] = None

    @validator('maturity_days')
    def validate_maturity_days(cls, v):
        if v <= 0 or v > 300:
            raise ValueError('Maturity days must be between 1 and 300')
        return v

class PlantingSessionInfo(BaseModel):
    """Planting session information"""
    planting_date: date
    expected_harvest_date: Optional[date] = None
    seed_rate_kg_per_hectare: Optional[float] = None
    row_spacing_cm: Optional[int] = None
    fertilizer_type: Optional[str] = None
    fertilizer_amount_kg_per_hectare: Optional[float] = None
    irrigation_method: Optional[str] = None
    notes: Optional[str] = None

class PredictionRequest(BaseModel):
    """Request for yield prediction - matches Kotlin PredictionRequest"""
    farm_id: int
    planting_session_id: int
    maize_variety: MaizeVarietyInfo
    planting_session: PlantingSessionInfo
    farm_location: Optional[FarmLocationData] = None
    soil_data: Optional[SoilDataRequest] = None
    weather_data: List[WeatherDataRequest] = []
    current_date: date = Field(default_factory=date.today)
    include_soil_data: bool = True
    include_weather_data: bool = True
    include_historical_yields: bool = True

    @validator('weather_data')
    def validate_weather_data(cls, v):
        if len(v) > 365:
            raise ValueError('Too many weather data points (max 365)')
        return v

class FactorImportance(BaseModel):
    """Factor importance for explainability - matches Kotlin FactorImportance"""
    factor: str
    importance: float = Field(..., ge=0.0, le=1.0)
    impact: str = Field(..., pattern="^(POSITIVE|NEGATIVE|NEUTRAL)$")
    description: Optional[str] = None

class PredictionResponse(BaseModel):
    """Response from yield prediction - matches Kotlin PredictionResult"""
    predicted_yield_tons_per_hectare: float = Field(..., ge=0.0)
    confidence_percentage: float = Field(..., ge=0.0, le=100.0)
    prediction_date: date = Field(default_factory=date.today)
    model_version: str
    features_used: List[str]
    important_factors: List[FactorImportance] = []
    risk_factors: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None

class ModelTrainingRequest(BaseModel):
    """Request for model retraining"""
    training_data_path: str
    model_config: Dict[str, Any] = {}
    validation_split: float = Field(default=0.2, ge=0.1, le=0.4)
    cross_validation_folds: int = Field(default=5, ge=3, le=10)
    performance_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

class ModelInfo(BaseModel):
    """Information about the current model"""
    version: str
    training_date: Optional[str] = None
    model_type: str
    accuracy_metrics: Dict[str, float] = {}
    feature_count: int
    last_updated: Optional[str] = None

class BatchPredictionRequest(BaseModel):
    """Request for batch predictions"""
    predictions: List[PredictionRequest]
    include_diagnostics: bool = False

class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    predictions: List[PredictionResponse]
    total_processed: int
    success_count: int
    error_count: int
    errors: List[str] = []
    processing_time_seconds: float

class ModelPerformanceMetrics(BaseModel):
    """Model performance metrics"""
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float # Root Mean Squared Error
    r2_score: float  # R-squared
    mape: float  # Mean Absolute Percentage Error
    accuracy_within_10_percent: float  # Percentage of predictions within 10% of actual

class FeatureImportanceResponse(BaseModel):
    """Global feature importance response"""
    feature_importance: List[FactorImportance]
    model_version: str
    generated_at: str

class ModelValidationRequest(BaseModel):
    """Request for model validation"""
    test_data_path: str
    validation_metrics: List[str] = ["mae", "rmse", "r2_score", "mape"]

class ModelValidationResponse(BaseModel):
    """Response for model validation"""
    metrics: ModelPerformanceMetrics
    validation_passed: bool
    threshold_metrics: Dict[str, float]
    recommendations: List[str] = []

# Historical yield data for training
class HistoricalYieldRecord(BaseModel):
    """Historical yield record for training"""
    farm_id: int
    planting_session_id: int
    harvest_date: date
    actual_yield_tons_per_hectare: float
    quality_rating: Optional[str] = None
    weather_conditions: List[WeatherDataRequest] = []
    soil_conditions: Optional[SoilDataRequest] = None
    planting_info: PlantingSessionInfo
    variety_info: MaizeVarietyInfo

class TrainingDataset(BaseModel):
    """Complete training dataset"""
    historical_yields: List[HistoricalYieldRecord]
    metadata: Dict[str, Any] = {}
    data_quality_score: Optional[float] = None

# Error response models
class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: str
    timestamp: str
    request_id: Optional[str] = None

class ValidationError(BaseModel):
    """Validation error details"""
    field: str
    message: str
    invalid_value: Any