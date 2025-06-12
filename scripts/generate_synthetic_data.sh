#!/bin/bash
# Generate synthetic training data for the ML model

echo "🌾 Generating synthetic training data for maize yield prediction..."

# Create data directories
mkdir -p data/training
mkdir -p logs

# Generate different sizes of datasets
echo "Generating small dataset (500 samples)..."
python3 generate_synthetic_data.py --samples 500 --output data/training/small_dataset.json

echo "Generating medium dataset (2000 samples)..."
python3 generate_synthetic_data.py --samples 2000 --output data/training/medium_dataset.json

echo "Generating large dataset (5000 samples)..."
python3 generate_synthetic_data.py --samples 5000 --output data/training/large_dataset.json

echo "Generating validation dataset (500 samples)..."
python3 generate_synthetic_data.py --samples 500 --output data/training/validation_dataset.json --years 2024

echo "✅ Synthetic data generation complete!"
echo "📊 Generated datasets:"
echo "  - Small: 500 samples (data/training/small_dataset.json)"
echo "  - Medium: 2000 samples (data/training/medium_dataset.json)"
echo "  - Large: 5000 samples (data/training/large_dataset.json)"
echo "  - Validation: 500 samples (data/training/validation_dataset.json)"
echo ""
echo "💡 You can now train your model using:"
echo "  python scripts/train_model.py --data-path data/training/medium_dataset.json"