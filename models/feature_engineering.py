"""
Feature Engineering for Maize Yield Prediction
Transforms raw agricultural data into ML-ready features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import date, datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from schemas.prediction_schemas import PredictionRequest

logger = logging.getLogger(__name__)

class FeatureEngineering:
    """
    Feature engineering for agricultural data
    """

    def __init__(self):
        self.label_encoders = {}
        self.scalers = {}
        self.feature_stats = {}

    def create_features(self, request: PredictionRequest) -> Dict[str, Any]:
        """
        Create features from prediction request
        """
        features = {}

        # Extract basic information
        planting_date = request.planting_session.planting_date
        current_date = request.current_date
        variety = request.maize_variety

        # === TEMPORAL FEATURES ===
        features.update(self._create_temporal_features(planting_date, current_date))

        # === VARIETY FEATURES ===
        features.update(self._create_variety_features(variety))

        # === LOCATION FEATURES ===
        if request.farm_location:
            features.update(self._create_location_features(request.farm_location))

        # === SOIL FEATURES ===
        if request.soil_data:
            features.update(self._create_soil_features(request.soil_data))

        # === WEATHER FEATURES ===
        if request.weather_data:
            features.update(self._create_weather_features(request.weather_data, planting_date))

        # === MANAGEMENT FEATURES ===
        features.update(self._create_management_features(request.planting_session))

        # === DERIVED FEATURES ===
        features.update(self._create_derived_features(features, variety, planting_date, current_date))

        return features

    def _create_temporal_features(self, planting_date: date, current_date: date) -> Dict[str, Any]:
        """Create time-based features"""
        features = {}

        # Days since planting
        days_since_planting = (current_date - planting_date).days
        features['days_since_planting'] = days_since_planting

        # Planting timing features
        features['planting_month'] = planting_date.month
        features['planting_day_of_year'] = planting_date.timetuple().tm_yday
        features['planting_week'] = planting_date.isocalendar()[1]

        # Seasonal features (cyclical encoding)
        day_of_year = planting_date.timetuple().tm_yday
        features['planting_season_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        features['planting_season_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

        # Current season
        current_day_of_year = current_date.timetuple().tm_yday
        features['current_season_sin'] = np.sin(2 * np.pi * current_day_of_year / 365.25)
        features['current_season_cos'] = np.cos(2 * np.pi * current_day_of_year / 365.25)

        # Season classification
        features['is_rainy_season'] = int(planting_date.month in [11, 12, 1, 2, 3, 4])  # Zimbabwe rainy season
        features['is_dry_season'] = int(planting_date.month in [5, 6, 7, 8, 9, 10])

        return features

    def _create_variety_features(self, variety) -> Dict[str, Any]:
        """Create maize variety features"""
        features = {}

        # Basic variety characteristics
        features['variety_maturity_days'] = variety.maturity_days
        features['variety_drought_resistance'] = int(variety.drought_resistance)

        # Temperature preferences
        features['variety_temp_min'] = variety.optimal_temperature_min or 18.0
        features['variety_temp_max'] = variety.optimal_temperature_max or 30.0
        features['variety_temp_range'] = features['variety_temp_max'] - features['variety_temp_min']
        features['variety_temp_midpoint'] = (features['variety_temp_max'] + features['variety_temp_min']) / 2

        # Maturity classification
        features['is_early_variety'] = int(variety.maturity_days < 100)
        features['is_medium_variety'] = int(100 <= variety.maturity_days <= 120)
        features['is_late_variety'] = int(variety.maturity_days > 120)

        # Variety name encoding (for known varieties)
        variety_mapping = {
            'ZM 523': 1, 'ZM 625': 2, 'ZM 309': 3, 'ZM 421': 4, 'ZM 701': 5,
            'PAN 4M-19': 6, 'SC Duma 43': 7, 'ZM 607': 8, 'Pioneer 30G19': 9, 'Dekalb DKC 80-40': 10
        }
        features['variety_encoded'] = variety_mapping.get(variety.name, 0)

        return features

    def _create_location_features(self, location) -> Dict[str, Any]:
        """Create farm location features"""
        features = {}

        features['latitude'] = location.latitude or -18.0  # Default to Zimbabwe center
        features['longitude'] = location.longitude or 31.0
        features['elevation'] = location.elevation or 1000.0

        # Derived location features
        features['abs_latitude'] = abs(features['latitude'])
        features['distance_from_equator'] = abs(features['latitude'])

        # Zimbabwe regional classification (approximate)
        lat, lon = features['latitude'], features['longitude']

        # Agro-ecological regions (simplified)
        if lat > -17.5:
            features['agroeco_region'] = 1  # Northern regions (I, IIa)
        elif lat > -19.0:
            features['agroeco_region'] = 2  # Central regions (IIb, III)
        else:
            features['agroeco_region'] = 3  # Southern regions (IV, V)

        # Elevation-based features
        features['is_high_altitude'] = int(features['elevation'] > 1200)
        features['is_low_altitude'] = int(features['elevation'] < 800)

        return features

    def _create_soil_features(self, soil_data) -> Dict[str, Any]:
        """Create soil-related features"""
        features = {}

        # Basic soil properties
        features['soil_ph'] = soil_data.ph_level or 6.5
        features['soil_organic_matter'] = soil_data.organic_matter_percentage or 3.0
        features['soil_nitrogen'] = soil_data.nitrogen_content or 1.5
        features['soil_phosphorus'] = soil_data.phosphorus_content or 1.0
        features['soil_potassium'] = soil_data.potassium_content or 2.0
        features['soil_moisture'] = soil_data.moisture_content or 25.0

        # Soil type encoding
        soil_type_mapping = {
            'sandy': 1, 'loamy': 2, 'clay': 3, 'silty': 4, 'sandy_loam': 5,
            'clay_loam': 6, 'silt_loam': 7, 'peat': 8, 'chalk': 9
        }
        features['soil_type_encoded'] = soil_type_mapping.get(soil_data.soil_type.lower(), 2)  # Default to loamy

        # Derived soil features
        features['soil_ph_optimal'] = int(5.5 <= features['soil_ph'] <= 7.0)  # Optimal pH for maize
        features['soil_ph_acidic'] = int(features['soil_ph'] < 5.5)
        features['soil_ph_alkaline'] = int(features['soil_ph'] > 7.0)

        # Nutrient status classification
        features['nitrogen_adequate'] = int(features['soil_nitrogen'] >= 1.5)
        features['phosphorus_adequate'] = int(features['soil_phosphorus'] >= 1.0)
        features['potassium_adequate'] = int(features['soil_potassium'] >= 2.0)

        # Soil fertility index (simplified)
        features['soil_fertility_index'] = (
            (features['soil_nitrogen'] / 3.0) * 0.4 +
            (features['soil_phosphorus'] / 2.0) * 0.3 +
            (features['soil_potassium'] / 3.0) * 0.2 +
            (features['soil_organic_matter'] / 5.0) * 0.1
        )

        # Soil moisture classification
        features['soil_moisture_adequate'] = int(20.0 <= features['soil_moisture'] <= 40.0)
        features['soil_moisture_low'] = int(features['soil_moisture'] < 20.0)
        features['soil_moisture_high'] = int(features['soil_moisture'] > 40.0)

        # Nutrient ratios
        if features['soil_phosphorus'] > 0:
            features['n_p_ratio'] = features['soil_nitrogen'] / features['soil_phosphorus']
        else:
            features['n_p_ratio'] = features['soil_nitrogen']

        if features['soil_potassium'] > 0:
            features['n_k_ratio'] = features['soil_nitrogen'] / features['soil_potassium']
        else:
            features['n_k_ratio'] = features['soil_nitrogen']

        return features

    def _create_weather_features(self, weather_data: List, planting_date: date) -> Dict[str, Any]:
        """Create comprehensive weather features"""
        features = {}

        if not weather_data:
            return self._get_default_weather_features()

        # Convert to DataFrame for easier manipulation
        weather_df = pd.DataFrame([
            {
                'date': w.date,
                'temp_min': w.min_temperature,
                'temp_max': w.max_temperature,
                'temp_avg': w.average_temperature,
                'rainfall': w.rainfall_mm or 0.0,
                'humidity': w.humidity_percentage,
                'wind_speed': w.wind_speed_kmh,
                'solar_radiation': w.solar_radiation
            }
            for w in weather_data
        ])

        # Sort by date
        weather_df = weather_df.sort_values('date')

        # === BASIC AGGREGATIONS ===
        # Temperature features
        temp_cols = ['temp_min', 'temp_max', 'temp_avg']
        for col in temp_cols:
            if weather_df[col].notna().any():
                features[f'{col}_mean'] = weather_df[col].mean()
                features[f'{col}_std'] = weather_df[col].std()
                features[f'{col}_min'] = weather_df[col].min()
                features[f'{col}_max'] = weather_df[col].max()

        # Rainfall features
        features['total_rainfall'] = weather_df['rainfall'].sum()
        features['avg_rainfall'] = weather_df['rainfall'].mean()
        features['rainfall_std'] = weather_df['rainfall'].std()
        features['max_daily_rainfall'] = weather_df['rainfall'].max()
        features['rainfall_days'] = (weather_df['rainfall'] > 1.0).sum()

        # Other weather features
        if weather_df['humidity'].notna().any():
            features['avg_humidity'] = weather_df['humidity'].mean()
            features['humidity_std'] = weather_df['humidity'].std()

        if weather_df['wind_speed'].notna().any():
            features['avg_wind_speed'] = weather_df['wind_speed'].mean()
            features['max_wind_speed'] = weather_df['wind_speed'].max()

        # === STRESS INDICATORS ===
        # Temperature stress
        if 'temp_max_mean' in features:
            features['heat_stress_days'] = (weather_df['temp_max'] > 35.0).sum()
            features['extreme_heat_days'] = (weather_df['temp_max'] > 40.0).sum()

        if 'temp_min_mean' in features:
            features['cold_stress_days'] = (weather_df['temp_min'] < 10.0).sum()
            features['frost_days'] = (weather_df['temp_min'] < 2.0).sum()

        # Drought indicators
        features['dry_spell_max'] = self._calculate_max_dry_spell(weather_df['rainfall'])
        features['drought_stress_days'] = (weather_df['rainfall'] == 0).sum()

        # Excess water indicators
        features['heavy_rain_days'] = (weather_df['rainfall'] > 25.0).sum()
        features['extreme_rain_days'] = (weather_df['rainfall'] > 50.0).sum()

        # === GROWTH STAGE SPECIFIC FEATURES ===
        # Split weather data by growth stages
        stage_features = self._create_stage_specific_weather(weather_df, planting_date)
        features.update(stage_features)

        # === WEATHER PATTERNS ===
        # Rainfall distribution
        features['rainfall_cv'] = features['rainfall_std'] / features['avg_rainfall'] if features['avg_rainfall'] > 0 else 0
        features['rainfall_skewness'] = weather_df['rainfall'].skew()

        # Temperature variability
        if 'temp_avg_mean' in features:
            features['temp_variability'] = features['temp_avg_std']
            features['diurnal_temp_range'] = features.get('temp_max_mean', 30) - features.get('temp_min_mean', 15)

        # === RECENT WEATHER (last 7 days) ===
        recent_weather = weather_df.tail(7)
        if not recent_weather.empty:
            features['recent_rainfall'] = recent_weather['rainfall'].sum()
            features['recent_avg_temp'] = recent_weather['temp_avg'].mean() if recent_weather['temp_avg'].notna().any() else 25.0
            features['recent_temp_stress'] = (recent_weather['temp_max'] > 35.0).sum()

        return features

    def _calculate_max_dry_spell(self, rainfall_series: pd.Series) -> int:
        """Calculate the maximum consecutive dry days"""
        dry_spell = 0
        max_dry_spell = 0

        for rainfall in rainfall_series:
            if rainfall <= 1.0:  # Consider <= 1mm as dry
                dry_spell += 1
                max_dry_spell = max(max_dry_spell, dry_spell)
            else:
                dry_spell = 0

        return max_dry_spell

    def _create_stage_specific_weather(self, weather_df: pd.DataFrame, planting_date: date) -> Dict[str, Any]:
        """Create weather features for specific growth stages"""
        features = {}

        # Define growth stages (days after planting)
        stages = {
            'establishment': (0, 30),      # Germination to V6
            'vegetative': (30, 60),        # V6 to VT
            'reproductive': (60, 90),      # VT to R3
            'grain_filling': (90, 120)     # R3 to R6
        }

        for stage_name, (start_day, end_day) in stages.items():
            start_date = planting_date + timedelta(days=start_day)
            end_date = planting_date + timedelta(days=end_day)

            # Filter weather data for this stage
            stage_weather = weather_df[
                (weather_df['date'] >= start_date) &
                (weather_df['date'] <= end_date)
            ]

            if not stage_weather.empty:
                # Temperature features for stage
                if stage_weather['temp_avg'].notna().any():
                    features[f'{stage_name}_avg_temp'] = stage_weather['temp_avg'].mean()
                    features[f'{stage_name}_temp_stress'] = (stage_weather['temp_max'] > 35.0).sum()

                # Rainfall features for stage
                features[f'{stage_name}_rainfall'] = stage_weather['rainfall'].sum()
                features[f'{stage_name}_rain_days'] = (stage_weather['rainfall'] > 1.0).sum()

                # Critical stage indicators
                if stage_name == 'reproductive':
                    features['pollination_heat_stress'] = (stage_weather['temp_max'] > 35.0).sum()
                    features['pollination_water_stress'] = (stage_weather['rainfall'] == 0).sum()

                if stage_name == 'grain_filling':
                    features['grain_fill_temp_optimal'] = ((stage_weather['temp_avg'] >= 20) &
                                                         (stage_weather['temp_avg'] <= 30)).sum()

        return features

    def _create_management_features(self, planting_session) -> Dict[str, Any]:
        """Create management practice features"""
        features = {}

        # Planting density and spacing
        features['seed_rate'] = planting_session.seed_rate_kg_per_hectare or 25.0
        features['row_spacing'] = planting_session.row_spacing_cm or 75.0

        # Calculate plant population (approximate)
        seeds_per_kg = 3500  # Average for maize
        area_per_plant = (features['row_spacing'] / 100) * 0.25  # Assuming 25cm in-row spacing
        features['plant_population'] = (features['seed_rate'] * seeds_per_kg) / (10000 / area_per_plant)

        # Planting density classification
        features['high_density'] = int(features['plant_population'] > 80000)
        features['medium_density'] = int(40000 <= features['plant_population'] <= 80000)
        features['low_density'] = int(features['plant_population'] < 40000)

        # Fertilizer management
        features['fertilizer_amount'] = planting_session.fertilizer_amount_kg_per_hectare or 150.0

        # Fertilizer type encoding
        fertilizer_mapping = {
            'npk': 1, 'urea': 2, 'can': 3, 'dap': 4, 'compound_d': 5,
            'ammonium_nitrate': 6, 'superphosphate': 7, 'organic': 8
        }
        fert_type = planting_session.fertilizer_type or 'npk'
        features['fertilizer_type_encoded'] = fertilizer_mapping.get(fert_type.lower(), 1)

        # Fertilizer adequacy
        features['fertilizer_adequate'] = int(features['fertilizer_amount'] >= 100.0)
        features['fertilizer_high'] = int(features['fertilizer_amount'] > 200.0)

        # Irrigation method
        irrigation_mapping = {
            'rainfed': 0, 'sprinkler': 1, 'drip': 2, 'furrow': 3, 'center_pivot': 4
        }
        irrigation = planting_session.irrigation_method or 'rainfed'
        features['irrigation_encoded'] = irrigation_mapping.get(irrigation.lower(), 0)
        features['is_irrigated'] = int(features['irrigation_encoded'] > 0)

        return features

    def _create_derived_features(self, features: Dict[str, Any], variety,
                                planting_date: date, current_date: date) -> Dict[str, Any]:
        """Create derived and interaction features"""
        derived = {}

        # Growth progress
        days_since_planting = features.get('days_since_planting', 0)
        maturity_days = variety.maturity_days

        derived['growth_progress'] = min(days_since_planting / maturity_days, 1.0)
        derived['days_to_harvest'] = max(0, maturity_days - days_since_planting)

        # Growth stage classification
        progress = derived['growth_progress']
        derived['stage_establishment'] = int(progress < 0.25)
        derived['stage_vegetative'] = int(0.25 <= progress < 0.5)
        derived['stage_reproductive'] = int(0.5 <= progress < 0.75)
        derived['stage_maturity'] = int(progress >= 0.75)

        # Temperature suitability
        if 'temp_avg_mean' in features:
            variety_temp_min = features.get('variety_temp_min', 18.0)
            variety_temp_max = features.get('variety_temp_max', 30.0)
            actual_temp = features['temp_avg_mean']

            # Temperature stress score
            if actual_temp < variety_temp_min:
                derived['temp_suitability'] = actual_temp / variety_temp_min
            elif actual_temp > variety_temp_max:
                derived['temp_suitability'] = variety_temp_max / actual_temp
            else:
                derived['temp_suitability'] = 1.0

        # Water balance features
        if 'total_rainfall' in features:
            # Simple water balance (rainfall vs evapotranspiration estimate)
            et_estimate = days_since_planting * 4.0  # Rough ET estimate (4mm/day)
            derived['water_balance'] = features['total_rainfall'] - et_estimate
            derived['water_stress_index'] = max(0, -derived['water_balance'] / et_estimate) if et_estimate > 0 else 0

        # Nutrient balance
        if 'soil_nitrogen' in features and 'fertilizer_amount' in features:
            total_n_available = features['soil_nitrogen'] + (features['fertilizer_amount'] * 0.2)  # Assume 20% N in fertilizer
            derived['nitrogen_sufficiency'] = min(total_n_available / 200.0, 1.0)  # 200 kg/ha is high

        # Stress combination index
        stress_factors = []
        if 'temp_suitability' in derived:
            stress_factors.append(1 - derived['temp_suitability'])
        if 'water_stress_index' in derived:
            stress_factors.append(derived['water_stress_index'])

        if stress_factors:
            derived['combined_stress_index'] = np.mean(stress_factors)

        # Yield potential modifier
        if 'soil_fertility_index' in features and 'temp_suitability' in derived:
            base_potential = 6.0  # Base yield potential (tons/ha)
            fertility_modifier = features['soil_fertility_index']
            temp_modifier = derived['temp_suitability']
            water_modifier = 1 - derived.get('water_stress_index', 0)

            derived['yield_potential'] = base_potential * fertility_modifier * temp_modifier * water_modifier

        return derived

    def _get_default_weather_features(self) -> Dict[str, Any]:
        """Return default weather features when no data is available"""
        return {
            'temp_avg_mean': 25.0, 'temp_avg_std': 5.0, 'temp_min_mean': 15.0,
            'temp_max_mean': 35.0, 'total_rainfall': 400.0, 'avg_rainfall': 5.0,
            'rainfall_std': 8.0, 'rainfall_days': 60, 'avg_humidity': 70.0,
            'avg_wind_speed': 10.0, 'heat_stress_days': 5, 'cold_stress_days': 2,
            'dry_spell_max': 7, 'drought_stress_days': 10, 'heavy_rain_days': 3,
            'rainfall_cv': 1.6, 'temp_variability': 5.0, 'diurnal_temp_range': 15.0,
            'recent_rainfall': 20.0, 'recent_avg_temp': 25.0, 'recent_temp_stress': 1
        }

    def create_training_features(self, historical_data: List[Dict]) -> pd.DataFrame:
        """
        Create features for training data
        """
        feature_list = []

        for record in historical_data:
            try:
                # Convert record to PredictionRequest format
                request = self._convert_to_prediction_request(record)
                features = self.create_features(request)

                # Add target variable
                features['yield'] = record['actual_yield_tons_per_hectare']

                feature_list.append(features)

            except Exception as e:
                logger.warning(f"Skipping record due to error: {str(e)}")
                continue

        if not feature_list:
            raise ValueError("No valid training records found")

        return pd.DataFrame(feature_list)

    def _convert_to_prediction_request(self, record: Dict) -> PredictionRequest:
        """Convert historical record to PredictionRequest format"""
        # This is a simplified conversion - you'd need to adapt based on your data format
        from schemas.prediction_schemas import (
            PredictionRequest, MaizeVarietyInfo, PlantingSessionInfo,
            SoilDataRequest, WeatherDataRequest, FarmLocationData
        )

        # Create variety info
        variety_info = MaizeVarietyInfo(
            id=record.get('variety_id', 1),
            name=record.get('variety_name', 'ZM 523'),
            maturity_days=record.get('maturity_days', 120),
            drought_resistance=record.get('drought_resistance', False)
        )

        # Create planting session info
        planting_info = PlantingSessionInfo(
            planting_date=record['planting_date'],
            seed_rate_kg_per_hectare=record.get('seed_rate', 25.0),
            row_spacing_cm=record.get('row_spacing', 75),
            fertilizer_amount_kg_per_hectare=record.get('fertilizer_amount', 150.0)
        )

        # Create soil data if available
        soil_data = None
        if 'soil_data' in record:
            soil_data = SoilDataRequest(**record['soil_data'])

        # Create weather data
        weather_data = []
        if 'weather_data' in record:
            weather_data = [WeatherDataRequest(**w) for w in record['weather_data']]

        # Create location data
        location_data = None
        if 'location' in record:
            location_data = FarmLocationData(**record['location'])

        return PredictionRequest(
            farm_id=record['farm_id'],
            planting_session_id=record['planting_session_id'],
            maize_variety=variety_info,
            planting_session=planting_info,
            farm_location=location_data,
            soil_data=soil_data,
            weather_data=weather_data,
            current_date=record['harvest_date']
        )

    def get_feature_names(self) -> List[str]:
        """Get list of all possible feature names"""
        # This would return all feature names that could be generated
        # You'd populate this based on the features created in your methods
        return [
            # Temporal features
            'days_since_planting', 'planting_month', 'planting_day_of_year',
            'planting_season_sin', 'planting_season_cos', 'is_rainy_season',

            # Variety features
            'variety_maturity_days', 'variety_drought_resistance', 'variety_temp_min',
            'variety_temp_max', 'is_early_variety', 'is_medium_variety', 'is_late_variety',

            # Location features
            'latitude', 'longitude', 'elevation', 'agroeco_region',

            # Soil features
            'soil_ph', 'soil_organic_matter', 'soil_nitrogen', 'soil_phosphorus',
            'soil_potassium', 'soil_moisture', 'soil_fertility_index',

            # Weather features
            'temp_avg_mean', 'total_rainfall', 'avg_humidity', 'heat_stress_days',
            'dry_spell_max', 'rainfall_cv', 'temp_variability',

            # Management features
            'seed_rate', 'fertilizer_amount', 'plant_population', 'is_irrigated',

            # Derived features
            'growth_progress', 'temp_suitability', 'water_balance', 'yield_potential'
        ]