# HarvestIQ ML Model Service

Machine Learning service for maize yield prediction in Zimbabwe.

## Features

- Ensemble ML models (Random Forest, XGBoost, LSTM)
- Explainable AI with SHAP
- Model versioning and management
- Comprehensive data validation
- RESTful API with FastAPI
- Docker support

## Quick Start

1. **Setup Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Start the Service:**
   ```bash
   ./scripts/start_service.sh
   ```

3. **Access API Documentation:**
   - Swagger UI: http://localhost:8001/docs
   - ReDoc: http://localhost:8001/redoc

## Development

1. **Install Development Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Tests:**
   ```bash
   pytest tests/
   ```

3. **Train Model:**
   ```bash
   python scripts/train_model.py --data-path ./data/training/training_data.json --create-synthetic
   ```

## API Endpoints

- `GET /` - Health check
- `POST /predict` - Generate yield prediction
- `GET /model/info` - Get model information
- `POST /model/retrain` - Trigger model retraining

## TODO

- [ ] Implement EnsembleYieldPredictor class
- [ ] Add feature engineering logic
- [ ] Implement model training pipeline
- [ ] Add comprehensive tests
- [ ] Set up CI/CD pipeline
- [ ] Add monitoring and alerting

## Architecture

```
harvestiq-ml-service/
├── main.py                 # FastAPI application
├── models/                 # ML model implementations
├── schemas/                # Pydantic schemas
├── utils/                  # Utility modules
├── scripts/                # Management scripts
├── tests/                  # Test suite
├── data/                   # Training data
├── storage/                # Model storage
└── config/                 # Configuration files
```
