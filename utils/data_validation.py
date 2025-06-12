"""
Data Validation Utilities for ML Model Service
Ensures data quality and consistency for training and prediction
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import date, datetime, timedelta
from schemas.prediction_schemas import PredictionRequest

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Comprehensive data validation for agricultural ML models
    """

    def __init__(self):
        self.validation_rules = self._setup_validation_rules()
        self.data_quality_thresholds = self._setup_quality_thresholds()

    def _setup_validation_rules(self) -> Dict[str, Dict]:
        """Setup validation rules for different data types"""
        return {
            'soil': {
                'ph_level': {'min': 3.0, 'max': 11.0, 'optimal_min': 5.5, 'optimal_max': 7.5},
                'organic_matter_percentage': {'min': 0.0, 'max': 15.0, 'optimal_min': 2.0, 'optimal_max': 6.0},
                'nitrogen_content': {'min': 0.0, 'max': 5.0, 'optimal_min': 1.0, 'optimal_max': 3.0},
                'phosphorus_content': {'min': 0.0, 'max': 3.0, 'optimal_min': 0.5, 'optimal_max': 2.0},
                'potassium_content': {'min': 0.0, 'max': 5.0, 'optimal_min': 1.5, 'optimal_max': 3.5},
                'moisture_content': {'min': 5.0, 'max': 60.0, 'optimal_min': 15.0, 'optimal_max': 35.0}
            },
            'weather': {
                'temperature': {'min': -10.0, 'max': 50.0, 'optimal_min': 15.0, 'optimal_max': 35.0},
                'rainfall_mm': {'min': 0.0, 'max': 200.0, 'daily_extreme': 100.0},
                'humidity_percentage': {'min': 10.0, 'max': 100.0, 'optimal_min': 40.0, 'optimal_max': 80.0},
                'wind_speed_kmh': {'min': 0.0, 'max': 150.0, 'extreme': 80.0}
            },
            'variety': {
                'maturity_days': {'min': 60, 'max': 200, 'typical_min': 80, 'typical_max': 140}
            },
            'management': {
                'seed_rate_kg_per_hectare': {'min': 10.0, 'max': 50.0, 'optimal_min': 20.0, 'optimal_max': 35.0},
                'fertilizer_amount_kg_per_hectare': {'min': 0.0, 'max': 500.0, 'typical_max': 300.0},
                'row_spacing_cm': {'min': 30, 'max': 150, 'typical_min': 60, 'typical_max': 100}
            },
            'location': {
                'latitude': {'min': -35.0, 'max': -10.0},  # Southern Africa range
                'longitude': {'min': 15.0, 'max': 40.0},   # Southern Africa range
                'elevation': {'min': 0.0, 'max': 3000.0, 'typical_max': 2000.0}
            },
            'yield': {
                'tons_per_hectare': {'min': 0.0, 'max': 15.0, 'typical_max': 12.0, 'realistic_max': 10.0}
            }
        }

    def _setup_quality_thresholds(self) -> Dict[str, float]:
        """Setup data quality thresholds"""
        return {
            'min_completeness': 0.8,  # 80% of critical fields must be present
            'max_missing_weather_days': 0.2,  # Max 20% missing weather data
            'min_weather_sequence_length': 30,  # Minimum 30 days of weather data
            'max_outlier_percentage': 0.05,  # Max 5% outliers allowed
            'min_yield_variance': 0.1,  # Minimum variance in yield data for training
            'max_duplicate_percentage': 0.1  # Max 10% duplicate records
        }

    def validate_prediction_request(self, request: PredictionRequest) -> bool:
        """
        Validate a prediction request
        """
        try:
            validation_results = []

            # Basic request validation
            validation_results.append(self._validate_basic_request(request))

            # Soil data validation
            if request.soil_data:
                validation_results.append(self._validate_soil_data(request.soil_data))

            # Weather data validation
            if request.weather_data:
                validation_results.append(self._validate_weather_data(request.weather_data))

            # Variety validation
            validation_results.append(self._validate_variety_data(request.maize_variety))

            # Management validation
            validation_results.append(self._validate_management_data(request.planting_session))

            # Location validation
            if request.farm_location:
                validation_results.append(self._validate_location_data(request.farm_location))

            # Check if all validations passed
            all_valid = all(validation_results)

            if not all_valid:
                logger.warning(f"Prediction request validation failed for farm {request.farm_id}")

            return all_valid

        except Exception as e:
            logger.error(f"Error validating prediction request: {str(e)}")
            return False

    def _validate_basic_request(self, request: PredictionRequest) -> bool:
        """Validate basic request structure"""
        errors = []

        # Check required fields
        if not request.farm_id or request.farm_id <= 0:
            errors.append("Invalid farm_id")

        if not request.planting_session_id or request.planting_session_id <= 0:
            errors.append("Invalid planting_session_id")

        # Validate dates
        planting_date = request.planting_session.planting_date
        current_date = request.current_date

        if planting_date > current_date:
            errors.append("Planting date cannot be in the future")

        # Check reasonable date range (within 2 years)
        date_diff = (current_date - planting_date).days
        if date_diff > 730:  # 2 years
            errors.append("Planting date is too far in the past")

        if errors:
            logger.warning(f"Basic validation errors: {errors}")
            return False

        return True

    def _validate_soil_data(self, soil_data) -> bool:
        """Validate soil data"""
        errors = []
        rules = self.validation_rules['soil']

        # pH validation
        if soil_data.ph_level is not None:
            ph = soil_data.ph_level
            if not (rules['ph_level']['min'] <= ph <= rules['ph_level']['max']):
                errors.append(f"pH level {ph} outside valid range")

        # Organic matter validation
        if soil_data.organic_matter_percentage is not None:
            om = soil_data.organic_matter_percentage
            if not (rules['organic_matter_percentage']['min'] <= om <= rules['organic_matter_percentage']['max']):
                errors.append(f"Organic matter {om}% outside valid range")

        # Nutrient validation
        nutrients = ['nitrogen_content', 'phosphorus_content', 'potassium_content']
        for nutrient in nutrients:
            value = getattr(soil_data, nutrient, None)
            if value is not None:
                rule = rules[nutrient]
                if not (rule['min'] <= value <= rule['max']):
                    errors.append(f"{nutrient} {value} outside valid range")

        # Moisture validation
        if soil_data.moisture_content is not None:
            moisture = soil_data.moisture_content
            if not (rules['moisture_content']['min'] <= moisture <= rules['moisture_content']['max']):
                errors.append(f"Soil moisture {moisture}% outside valid range")

        # Sample date validation
        if soil_data.sample_date:
            days_old = (date.today() - soil_data.sample_date).days
            if days_old > 365:  # Soil data older than 1 year
                logger.warning(f"Soil data is {days_old} days old - may be outdated")

        if errors:
            logger.warning(f"Soil validation errors: {errors}")
            return False

        return True

    def _validate_weather_data(self, weather_data: List) -> bool:
        """Validate weather data"""
        if not weather_data:
            logger.warning("No weather data provided")
            return True  # Weather data is optional

        errors = []
        rules = self.validation_rules['weather']

        # Check data completeness
        total_days = len(weather_data)
        if total_days < self.data_quality_thresholds['min_weather_sequence_length']:
            errors.append(f"Weather sequence too short: {total_days} days")

        # Validate individual weather records
        valid_records = 0
        for i, weather in enumerate(weather_data):
            record_valid = True

            # Temperature validation
            for temp_field in ['min_temperature', 'max_temperature', 'average_temperature']:
                temp = getattr(weather, temp_field, None)
                if temp is not None:
                    if not (rules['temperature']['min'] <= temp <= rules['temperature']['max']):
                        errors.append(f"Day {i}: {temp_field} {temp}°C outside valid range")
                        record_valid = False

            # Temperature consistency check
            if (weather.min_temperature is not None and weather.max_temperature is not None):
                if weather.min_temperature > weather.max_temperature:
                    errors.append(f"Day {i}: Min temperature > Max temperature")
                    record_valid = False

            # Rainfall validation
            if weather.rainfall_mm is not None:
                rainfall = weather.rainfall_mm
                if not (rules['rainfall_mm']['min'] <= rainfall <= rules['rainfall_mm']['max']):
                    errors.append(f"Day {i}: Rainfall {rainfall}mm outside valid range")
                    record_valid = False

                # Check for extreme rainfall
                if rainfall > rules['rainfall_mm']['daily_extreme']:
                    logger.warning(f"Day {i}: Extreme rainfall detected: {rainfall}mm")

            # Humidity validation
            if weather.humidity_percentage is not None:
                humidity = weather.humidity_percentage
                if not (rules['humidity_percentage']['min'] <= humidity <= rules['humidity_percentage']['max']):
                    errors.append(f"Day {i}: Humidity {humidity}% outside valid range")
                    record_valid = False

            # Wind speed validation
            if weather.wind_speed_kmh is not None:
                wind = weather.wind_speed_kmh
                if not (rules['wind_speed_kmh']['min'] <= wind <= rules['wind_speed_kmh']['max']):
                    errors.append(f"Day {i}: Wind speed {wind} km/h outside valid range")
                    record_valid = False

                if wind > rules['wind_speed_kmh']['extreme']:
                    logger.warning(f"Day {i}: Extreme wind speed: {wind} km/h")

            if record_valid:
                valid_records += 1

        # Check data quality
        completeness = valid_records / total_days if total_days > 0 else 0
        if completeness < self.data_quality_thresholds['min_completeness']:
            errors.append(f"Weather data completeness too low: {completeness:.2%}")

        # Check for data gaps
        if len(weather_data) > 1:
            dates = [w.date for w in weather_data]
            date_gaps = self._detect_date_gaps(dates)
            if date_gaps > self.data_quality_thresholds['max_missing_weather_days'] * len(dates):
                errors.append(f"Too many gaps in weather data: {date_gaps} missing days")

        if errors:
            logger.warning(f"Weather validation errors: {errors}")
            return False

        return True

    def _detect_date_gaps(self, dates: List[date]) -> int:
        """Detect gaps in date sequence"""
        if len(dates) < 2:
            return 0

        sorted_dates = sorted(dates)
        total_gaps = 0

        for i in range(1, len(sorted_dates)):
            expected_date = sorted_dates[i-1] + timedelta(days=1)
            if sorted_dates[i] > expected_date:
                gap_days = (sorted_dates[i] - expected_date).days
                total_gaps += gap_days

        return total_gaps

    def _validate_variety_data(self, variety) -> bool:
        """Validate maize variety data"""
        errors = []
        rules = self.validation_rules['variety']

        # Maturity days validation
        maturity = variety.maturity_days
        if not (rules['maturity_days']['min'] <= maturity <= rules['maturity_days']['max']):
            errors.append(f"Maturity days {maturity} outside valid range")

        # Temperature range validation
        if variety.optimal_temperature_min and variety.optimal_temperature_max:
            temp_min = variety.optimal_temperature_min
            temp_max = variety.optimal_temperature_max

            temp_rules = self.validation_rules['weather']['temperature']
            if not (temp_rules['min'] <= temp_min <= temp_rules['max']):
                errors.append(f"Optimal temperature min {temp_min}°C outside valid range")

            if not (temp_rules['min'] <= temp_max <= temp_rules['max']):
                errors.append(f"Optimal temperature max {temp_max}°C outside valid range")

            if temp_min >= temp_max:
                errors.append("Optimal temperature min >= max")

        # Variety name validation (optional - check against known varieties)
        known_varieties = [
            'ZM 523', 'ZM 625', 'ZM 309', 'ZM 421', 'ZM 701',
            'PAN 4M-19', 'SC Duma 43', 'ZM 607', 'Pioneer 30G19', 'Dekalb DKC 80-40'
        ]

        if variety.name not in known_varieties:
            logger.info(f"Unknown variety: {variety.name}")

        if errors:
            logger.warning(f"Variety validation errors: {errors}")
            return False

        return True

    def _validate_management_data(self, planting_session) -> bool:
        """Validate management/planting data"""
        errors = []
        rules = self.validation_rules['management']

        # Seed rate validation
        if planting_session.seed_rate_kg_per_hectare:
            seed_rate = planting_session.seed_rate_kg_per_hectare
            if not (rules['seed_rate_kg_per_hectare']['min'] <= seed_rate <= rules['seed_rate_kg_per_hectare']['max']):
                errors.append(f"Seed rate {seed_rate} kg/ha outside valid range")

        # Row spacing validation
        if planting_session.row_spacing_cm:
            spacing = planting_session.row_spacing_cm
            if not (rules['row_spacing_cm']['min'] <= spacing <= rules['row_spacing_cm']['max']):
                errors.append(f"Row spacing {spacing} cm outside valid range")

        # Fertilizer validation
        if planting_session.fertilizer_amount_kg_per_hectare:
            fert_amount = planting_session.fertilizer_amount_kg_per_hectare
            if not (rules['fertilizer_amount_kg_per_hectare']['min'] <= fert_amount <= rules['fertilizer_amount_kg_per_hectare']['max']):
                errors.append(f"Fertilizer amount {fert_amount} kg/ha outside valid range")

        # Date consistency (if expected harvest date is provided)
        if planting_session.expected_harvest_date:
            planting_date = planting_session.planting_date
            harvest_date = planting_session.expected_harvest_date

            if harvest_date <= planting_date:
                errors.append("Expected harvest date must be after planting date")

            growing_period = (harvest_date - planting_date).days
            if growing_period < 60 or growing_period > 200:
                errors.append(f"Growing period {growing_period} days seems unrealistic")

        if errors:
            logger.warning(f"Management validation errors: {errors}")
            return False

        return True

    def _validate_location_data(self, location) -> bool:
        """Validate farm location data"""
        errors = []
        rules = self.validation_rules['location']

        # Latitude validation
        if location.latitude:
            lat = location.latitude
            if not (rules['latitude']['min'] <= lat <= rules['latitude']['max']):
                errors.append(f"Latitude {lat} outside Southern Africa range")

        # Longitude validation
        if location.longitude:
            lon = location.longitude
            if not (rules['longitude']['min'] <= lon <= rules['longitude']['max']):
                errors.append(f"Longitude {lon} outside Southern Africa range")

        # Elevation validation
        if location.elevation:
            elev = location.elevation
            if not (rules['elevation']['min'] <= elev <= rules['elevation']['max']):
                errors.append(f"Elevation {elev}m outside valid range")

        if errors:
            logger.warning(f"Location validation errors: {errors}")
            return False

        return True

    def validate_training_data(self, training_data: List[Dict]) -> bool:
        """
        Validate training dataset quality
        """
        try:
            logger.info(f"Validating training dataset with {len(training_data)} records")

            if len(training_data) < 50:
                logger.error("Training dataset too small (minimum 50 records required)")
                return False

            validation_results = []

            # Individual record validation
            validation_results.append(self._validate_training_records(training_data))

            # Dataset-level validation
            validation_results.append(self._validate_dataset_quality(training_data))

            # Yield distribution validation
            validation_results.append(self._validate_yield_distribution(training_data))

            # Feature coverage validation
            validation_results.append(self._validate_feature_coverage(training_data))

            all_valid = all(validation_results)

            if all_valid:
                logger.info("Training data validation passed")
            else:
                logger.error("Training data validation failed")

            return all_valid

        except Exception as e:
            logger.error(f"Error validating training data: {str(e)}")
            return False

    def _validate_training_records(self, training_data: List[Dict]) -> bool:
        """Validate individual training records"""
        valid_records = 0
        total_records = len(training_data)

        for i, record in enumerate(training_data):
            try:
                # Check required fields
                required_fields = ['farm_id', 'planting_session_id', 'actual_yield_tons_per_hectare',
                                 'planting_date', 'harvest_date']

                if not all(field in record for field in required_fields):
                    continue

                # Validate yield
                yield_value = record['actual_yield_tons_per_hectare']
                yield_rules = self.validation_rules['yield']['tons_per_hectare']

                if not (yield_rules['min'] <= yield_value <= yield_rules['max']):
                    continue

                # Validate dates
                planting_date = pd.to_datetime(record['planting_date']).date()
                harvest_date = pd.to_datetime(record['harvest_date']).date()

                if harvest_date <= planting_date:
                    continue

                growing_period = (harvest_date - planting_date).days
                if growing_period < 60 or growing_period > 200:
                    continue

                valid_records += 1

            except Exception as e:
                logger.warning(f"Error validating record {i}: {str(e)}")
                continue

        completeness = valid_records / total_records
        logger.info(f"Training data completeness: {completeness:.2%} ({valid_records}/{total_records})")

        return completeness >= self.data_quality_thresholds['min_completeness']

    def _validate_dataset_quality(self, training_data: List[Dict]) -> bool:
        """Validate overall dataset quality"""
        # Check for duplicates
        seen_records = set()
        duplicates = 0

        for record in training_data:
            # Create a signature for duplicate detection
            signature = (
                record.get('farm_id'),
                record.get('planting_date'),
                record.get('harvest_date')
            )

            if signature in seen_records:
                duplicates += 1
            else:
                seen_records.add(signature)

        duplicate_rate = duplicates / len(training_data)
        if duplicate_rate > self.data_quality_thresholds['max_duplicate_percentage']:
            logger.warning(f"High duplicate rate: {duplicate_rate:.2%}")
            return False

        # Check data distribution across time
        dates = []
        for record in training_data:
            try:
                planting_date = pd.to_datetime(record['planting_date']).date()
                dates.append(planting_date)
            except:
                continue

        if dates:
            date_range = max(dates) - min(dates)
            if date_range.days < 365:
                logger.warning(f"Limited temporal coverage: {date_range.days} days")

        return True

    def _validate_yield_distribution(self, training_data: List[Dict]) -> bool:
        """Validate yield value distribution"""
        yields = []

        for record in training_data:
            try:
                yield_value = record['actual_yield_tons_per_hectare']
                if 0 <= yield_value <= 15:  # Reasonable range
                    yields.append(yield_value)
            except:
                continue

        if not yields:
            logger.error("No valid yield values found")
            return False

        yields_array = np.array(yields)

        # Check variance
        yield_variance = np.var(yields_array)
        if yield_variance < self.data_quality_thresholds['min_yield_variance']:
            logger.warning(f"Low yield variance: {yield_variance:.3f}")
            return False

        # Check for outliers
        q1, q3 = np.percentile(yields_array, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = np.sum((yields_array < lower_bound) | (yields_array > upper_bound))
        outlier_rate = outliers / len(yields_array)

        if outlier_rate > self.data_quality_thresholds['max_outlier_percentage']:
            logger.warning(f"High outlier rate: {outlier_rate:.2%}")

        # Log distribution statistics
        logger.info(f"Yield distribution - Mean: {np.mean(yields_array):.2f}, "
                   f"Std: {np.std(yields_array):.2f}, "
                   f"Range: {np.min(yields_array):.2f}-{np.max(yields_array):.2f}")

        return True

    def _validate_feature_coverage(self, training_data: List[Dict]) -> bool:
        """Validate feature coverage across the dataset"""
        feature_coverage = {}

        # Check coverage of important features
        important_features = [
            'soil_data', 'weather_data', 'variety_name', 'location',
            'seed_rate', 'fertilizer_amount', 'irrigation_method'
        ]

        for feature in important_features:
            present_count = 0
            for record in training_data:
                if feature in record and record[feature] is not None:
                    if isinstance(record[feature], (list, dict)):
                        if record[feature]:  # Non-empty list/dict
                            present_count += 1
                    else:
                        present_count += 1

            coverage = present_count / len(training_data)
            feature_coverage[feature] = coverage

        # Log feature coverage
        for feature, coverage in feature_coverage.items():
            logger.info(f"{feature} coverage: {coverage:.2%}")

        # Check if critical features have sufficient coverage
        critical_features = ['variety_name', 'weather_data']
        for feature in critical_features:
            if feature_coverage.get(feature, 0) < 0.5:  # 50% minimum
                logger.warning(f"Low coverage for critical feature {feature}: {feature_coverage[feature]:.2%}")
                return False

        return True

    def generate_data_quality_report(self, data: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        report = {
            'total_records': len(data),
            'timestamp': datetime.now().isoformat(),
            'validation_summary': {},
            'feature_coverage': {},
            'data_distribution': {},
            'quality_issues': []
        }

        try:
            # Basic validation
            valid_records = 0
            for record in data:
                if self._is_valid_record(record):
                    valid_records += 1

            report['validation_summary']['valid_records'] = valid_records
            report['validation_summary']['completion_rate'] = valid_records / len(data)

            # Feature coverage analysis
            features = ['soil_data', 'weather_data', 'variety_name', 'location']
            for feature in features:
                coverage = sum(1 for record in data if record.get(feature)) / len(data)
                report['feature_coverage'][feature] = coverage

            # Yield distribution
            yields = [r['actual_yield_tons_per_hectare'] for r in data
                     if 'actual_yield_tons_per_hectare' in r and
                     isinstance(r['actual_yield_tons_per_hectare'], (int, float))]

            if yields:
                report['data_distribution']['yield'] = {
                    'count': len(yields),
                    'mean': float(np.mean(yields)),
                    'std': float(np.std(yields)),
                    'min': float(np.min(yields)),
                    'max': float(np.max(yields)),
                    'percentiles': {
                        '25': float(np.percentile(yields, 25)),
                        '50': float(np.percentile(yields, 50)),
                        '75': float(np.percentile(yields, 75))
                    }
                }

            # Identify quality issues
            if report['validation_summary']['completion_rate'] < 0.8:
                report['quality_issues'].append("Low data completion rate")

            if any(coverage < 0.5 for coverage in report['feature_coverage'].values()):
                report['quality_issues'].append("Poor feature coverage")

            # Add recommendations
            report['recommendations'] = self._generate_quality_recommendations(report)

        except Exception as e:
            logger.error(f"Error generating data quality report: {str(e)}")
            report['error'] = str(e)

        return report

    def _is_valid_record(self, record: Dict) -> bool:
        """Check if a record is valid"""
        required_fields = ['farm_id', 'actual_yield_tons_per_hectare', 'planting_date']
        return all(field in record and record[field] is not None for field in required_fields)

    def _generate_quality_recommendations(self, report: Dict) -> List[str]:
        """Generate data quality improvement recommendations"""
        recommendations = []

        completion_rate = report['validation_summary'].get('completion_rate', 0)
        if completion_rate < 0.8:
            recommendations.append("Improve data collection to increase completion rate above 80%")

        for feature, coverage in report['feature_coverage'].items():
            if coverage < 0.5:
                recommendations.append(f"Increase {feature} coverage (currently {coverage:.1%})")

        if 'yield' in report['data_distribution']:
            yield_data = report['data_distribution']['yield']
            if yield_data['std'] < 0.5:
                recommendations.append("Consider collecting data from more diverse farming conditions")

        if not recommendations:
            recommendations.append("Data quality is good - no specific improvements needed")

        return recommendations