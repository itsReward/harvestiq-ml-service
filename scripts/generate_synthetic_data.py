#!/usr/bin/env python3
"""
Synthetic Training Data Generator for Maize Yield Prediction
Creates realistic agricultural data for Zimbabwe's farming conditions
"""

import json
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
import argparse
import random
from typing import Dict, List, Any

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)


class SyntheticDataGenerator:
    def __init__(self):
        # Zimbabwe agricultural zones and their characteristics
        self.agro_zones = {
            'I': {
                'rainfall_range': (1000, 1400),
                'temp_range': (18, 28),
                'base_yield': 6.5,
                'latitude_range': (-15.5, -17.0),
                'longitude_range': (28.0, 32.0)
            },
            'IIa': {
                'rainfall_range': (800, 1000),
                'temp_range': (19, 30),
                'base_yield': 5.8,
                'latitude_range': (-17.0, -18.5),
                'longitude_range': (28.0, 32.0)
            },
            'IIb': {
                'rainfall_range': (650, 800),
                'temp_range': (20, 32),
                'base_yield': 4.5,
                'latitude_range': (-18.5, -20.0),
                'longitude_range': (26.0, 33.0)
            },
            'III': {
                'rainfall_range': (500, 650),
                'temp_range': (18, 30),
                'base_yield': 3.2,
                'latitude_range': (-20.0, -21.5),
                'longitude_range': (25.0, 33.0)
            },
            'IV': {
                'rainfall_range': (450, 650),
                'temp_range': (22, 35),
                'base_yield': 2.1,
                'latitude_range': (-19.0, -22.0),
                'longitude_range': (25.0, 30.0)
            },
            'V': {
                'rainfall_range': (300, 500),
                'temp_range': (25, 40),
                'base_yield': 1.2,
                'latitude_range': (-21.0, -22.5),
                'longitude_range': (26.0, 31.0)
            }
        }

        # Maize varieties commonly grown in Zimbabwe
        self.maize_varieties = [
            {
                'id': 1, 'name': 'ZM 523', 'maturity_days': 120,
                'optimal_temp_min': 18.0, 'optimal_temp_max': 30.0,
                'drought_resistance': True, 'yield_potential': 1.15,
                'disease_resistance': 'Gray Leaf Spot, Northern Corn Leaf Blight'
            },
            {
                'id': 2, 'name': 'ZM 625', 'maturity_days': 125,
                'optimal_temp_min': 20.0, 'optimal_temp_max': 32.0,
                'drought_resistance': False, 'yield_potential': 1.25,
                'disease_resistance': 'Maize Streak Virus, Common Rust'
            },
            {
                'id': 3, 'name': 'ZM 309', 'maturity_days': 90,
                'optimal_temp_min': 18.0, 'optimal_temp_max': 28.0,
                'drought_resistance': True, 'yield_potential': 1.05,
                'disease_resistance': 'Gray Leaf Spot'
            },
            {
                'id': 4, 'name': 'ZM 421', 'maturity_days': 105,
                'optimal_temp_min': 19.0, 'optimal_temp_max': 29.0,
                'drought_resistance': False, 'yield_potential': 1.10,
                'disease_resistance': 'Northern Corn Leaf Blight, Common Rust'
            },
            {
                'id': 5, 'name': 'Pioneer 30G19', 'maturity_days': 118,
                'optimal_temp_min': 20.0, 'optimal_temp_max': 32.0,
                'drought_resistance': True, 'yield_potential': 1.30,
                'disease_resistance': 'Gray Leaf Spot, Northern Corn Leaf Blight'
            },
            {
                'id': 6, 'name': 'SC Duma 43', 'maturity_days': 130,
                'optimal_temp_min': 19.0, 'optimal_temp_max': 31.0,
                'drought_resistance': False, 'yield_potential': 1.35,
                'disease_resistance': 'Northern Corn Leaf Blight, Common Rust'
            }
        ]

        # Soil types common in Zimbabwe
        self.soil_types = {
            'sandy': {'drainage': 'excellent', 'fertility': 0.7, 'water_retention': 0.4},
            'loamy': {'drainage': 'good', 'fertility': 1.0, 'water_retention': 0.8},
            'clay': {'drainage': 'poor', 'fertility': 0.9, 'water_retention': 1.2},
            'sandy_loam': {'drainage': 'good', 'fertility': 0.85, 'water_retention': 0.7},
            'clay_loam': {'drainage': 'moderate', 'fertility': 1.1, 'water_retention': 1.0}
        }

        # Typical farming seasons in Zimbabwe
        self.planting_seasons = [
            {'start_month': 10, 'end_month': 12, 'probability': 0.7, 'name': 'early'},
            {'start_month': 11, 'end_month': 1, 'probability': 0.25, 'name': 'normal'},
            {'start_month': 12, 'end_month': 2, 'probability': 0.05, 'name': 'late'}
        ]

    def generate_farm_location(self, agro_zone: str) -> Dict[str, float]:
        """Generate realistic farm coordinates within specified agro-zone"""
        zone_data = self.agro_zones[agro_zone]

        return {
            'latitude': np.random.uniform(*zone_data['latitude_range']),
            'longitude': np.random.uniform(*zone_data['longitude_range']),
            'elevation': np.random.uniform(400, 1800)  # Zimbabwe elevation range
        }

    def generate_soil_data(self, soil_type: str, sample_date: str) -> Dict[str, Any]:
        """Generate realistic soil data based on soil type"""
        base_properties = self.soil_types[soil_type]

        # Generate soil chemical properties with realistic correlations
        if soil_type == 'sandy':
            ph_mean, ph_std = 6.2, 0.8
            om_mean, om_std = 1.8, 0.6
            n_mean, n_std = 0.8, 0.3
        elif soil_type == 'clay':
            ph_mean, ph_std = 6.8, 0.6
            om_mean, om_std = 3.2, 1.0
            n_mean, n_std = 1.6, 0.5
        else:  # loamy soils
            ph_mean, ph_std = 6.5, 0.5
            om_mean, om_std = 2.5, 0.8
            n_mean, n_std = 1.2, 0.4

        # Generate correlated P and K values
        nitrogen = max(0.1, np.random.normal(n_mean, n_std))
        phosphorus = max(0.1, np.random.normal(nitrogen * 0.7, 0.3))
        potassium = max(0.1, np.random.normal(nitrogen * 1.5, 0.6))

        return {
            'soil_type': soil_type,
            'ph_level': max(4.0, min(8.5, np.random.normal(ph_mean, ph_std))),
            'organic_matter_percentage': max(0.5, np.random.normal(om_mean, om_std)),
            'nitrogen_content': nitrogen,
            'phosphorus_content': phosphorus,
            'potassium_content': potassium,
            'moisture_content': max(5.0, min(50.0, np.random.normal(25.0, 8.0))),
            'sample_date': sample_date
        }

    def generate_weather_sequence(self, start_date: datetime, days: int,
                                  agro_zone: str, year: int) -> List[Dict[str, Any]]:
        """Generate realistic weather sequence for growing period"""
        zone_data = self.agro_zones[agro_zone]
        weather_data = []

        # Base temperature and rainfall patterns
        base_temp = np.random.uniform(*zone_data['temp_range'])
        annual_rainfall = np.random.uniform(*zone_data['rainfall_range'])

        # Seasonal patterns
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            day_of_year = current_date.timetuple().tm_yday

            # Temperature variation (seasonal + daily variation)
            seasonal_temp_factor = np.sin(2 * np.pi * (day_of_year - 80) / 365) * 8
            daily_temp = base_temp + seasonal_temp_factor + np.random.normal(0, 3)

            # Realistic temperature constraints
            min_temp = daily_temp - np.random.uniform(8, 15)
            max_temp = daily_temp + np.random.uniform(10, 18)

            # Rainfall patterns (Zimbabwe rainy season: Nov-Mar)
            month = current_date.month
            if month in [11, 12, 1, 2, 3]:  # Rainy season
                rain_prob = 0.4 if month in [12, 1, 2] else 0.25
                if np.random.random() < rain_prob:
                    # Log-normal distribution for rainfall amounts
                    rainfall = np.random.lognormal(2.0, 1.2)
                    rainfall = min(rainfall, 120.0)  # Cap extreme values
                else:
                    rainfall = 0.0
            else:  # Dry season
                rain_prob = 0.05
                rainfall = np.random.exponential(2.0) if np.random.random() < rain_prob else 0.0

            # Other weather parameters
            humidity = np.clip(np.random.normal(65, 15), 20, 95)
            wind_speed = np.clip(np.random.normal(12, 5), 2, 30)
            solar_radiation = np.clip(np.random.normal(18, 4), 8, 30)

            weather_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'min_temperature': round(min_temp, 1),
                'max_temperature': round(max_temp, 1),
                'average_temperature': round(daily_temp, 1),
                'rainfall_mm': round(rainfall, 1),
                'humidity_percentage': round(humidity, 1),
                'wind_speed_kmh': round(wind_speed, 1),
                'solar_radiation': round(solar_radiation, 1)
            })

        return weather_data

    def calculate_yield(self, variety: Dict, agro_zone: str, soil_data: Dict,
                        weather_data: List[Dict], management: Dict) -> float:
        """Calculate realistic yield based on all factors"""

        # Base yield from agro-zone
        base_yield = self.agro_zones[agro_zone]['base_yield']

        # Variety factor
        variety_factor = variety['yield_potential']

        # Soil fertility factor
        soil_type_data = self.soil_types[soil_data['soil_type']]
        soil_factor = soil_type_data['fertility']

        # pH adjustment
        optimal_ph = 6.5
        ph_factor = max(0.7, 1.0 - abs(soil_data['ph_level'] - optimal_ph) * 0.15)

        # Organic matter factor
        om_factor = min(1.3, 0.8 + soil_data['organic_matter_percentage'] * 0.08)

        # Nutrient factor
        nutrient_factor = min(1.2, 0.7 + soil_data['nitrogen_content'] * 0.25)

        # Weather factors
        total_rainfall = sum(w['rainfall_mm'] for w in weather_data)
        avg_temp = np.mean([w['average_temperature'] for w in weather_data])

        # Rainfall suitability
        optimal_rainfall = 600  # mm for full season
        if total_rainfall < 300:
            rainfall_factor = 0.3
        elif total_rainfall < 500:
            rainfall_factor = 0.3 + (total_rainfall - 300) * 0.002
        elif total_rainfall < 800:
            rainfall_factor = 0.7 + (total_rainfall - 500) * 0.001
        else:
            rainfall_factor = max(0.8, 1.0 - (total_rainfall - 800) * 0.0005)

        # Temperature suitability
        variety_temp_optimal = (variety['optimal_temp_min'] + variety['optimal_temp_max']) / 2
        temp_factor = max(0.6, 1.0 - abs(avg_temp - variety_temp_optimal) * 0.03)

        # Drought stress adjustment
        if not variety['drought_resistance'] and total_rainfall < 450:
            drought_penalty = 0.7
        else:
            drought_penalty = 1.0

        # Management factors
        seed_rate_factor = 1.0
        if management['seed_rate_kg_per_hectare'] < 20:
            seed_rate_factor = 0.9
        elif management['seed_rate_kg_per_hectare'] > 35:
            seed_rate_factor = 0.95

        fertilizer_factor = min(1.2, 0.8 + management['fertilizer_amount_kg_per_hectare'] / 300)

        # Heat stress during critical periods
        heat_stress_days = sum(1 for w in weather_data[45:75] if w['max_temperature'] > 35)
        heat_stress_factor = max(0.8, 1.0 - heat_stress_days * 0.02)

        # Calculate final yield
        final_yield = (base_yield * variety_factor * soil_factor * ph_factor *
                       om_factor * nutrient_factor * rainfall_factor * temp_factor *
                       drought_penalty * seed_rate_factor * fertilizer_factor *
                       heat_stress_factor)

        # Add some random variation
        final_yield *= np.random.normal(1.0, 0.1)

        # Ensure realistic bounds
        final_yield = max(0.3, min(12.0, final_yield))

        return round(final_yield, 2)

    def generate_sample(self, sample_id: int, year: int = 2023) -> Dict[str, Any]:
        """Generate a single training sample"""

        # Select random agro-zone
        agro_zone = np.random.choice(list(self.agro_zones.keys()))

        # Select planting season
        season = np.random.choice(self.planting_seasons, p=[s['probability'] for s in self.planting_seasons])

        # Generate planting date
        if season['start_month'] <= season['end_month']:
            planting_month = np.random.randint(season['start_month'], season['end_month'] + 1)
            planting_year = year
        else:  # Season crosses year boundary
            if np.random.random() < 0.7:
                planting_month = np.random.randint(season['start_month'], 13)
                planting_year = year
            else:
                planting_month = np.random.randint(1, season['end_month'] + 1)
                planting_year = year + 1

        planting_day = np.random.randint(1, 29)
        planting_date = datetime(planting_year, planting_month, planting_day)

        # Select variety
        variety = np.random.choice(self.maize_varieties)

        # Calculate harvest date
        harvest_date = planting_date + timedelta(days=variety['maturity_days'])

        # Generate location
        location = self.generate_farm_location(agro_zone)

        # Generate soil data
        soil_type = np.random.choice(list(self.soil_types.keys()),
                                     p=[0.25, 0.35, 0.15, 0.15, 0.10])
        soil_data = self.generate_soil_data(soil_type, planting_date.strftime('%Y-%m-%d'))

        # Generate management practices
        management = {
            'seed_rate_kg_per_hectare': np.clip(np.random.normal(25, 5), 15, 40),
            'row_spacing_cm': np.random.choice([75, 90], p=[0.7, 0.3]),
            'fertilizer_type': np.random.choice(['NPK', 'Urea', 'CAN', 'Compound D'],
                                                p=[0.4, 0.25, 0.2, 0.15]),
            'fertilizer_amount_kg_per_hectare': np.clip(np.random.normal(150, 60), 0, 400),
            'irrigation_method': np.random.choice(['rainfed', 'sprinkler', 'drip'],
                                                  p=[0.8, 0.15, 0.05])
        }

        # Generate weather data for growing season
        weather_data = self.generate_weather_sequence(
            planting_date, variety['maturity_days'], agro_zone, planting_year
        )

        # Calculate yield
        actual_yield = self.calculate_yield(variety, agro_zone, soil_data, weather_data, management)

        # Create the complete sample
        sample = {
            'farm_id': sample_id,
            'planting_session_id': sample_id,
            'planting_date': planting_date.strftime('%Y-%m-%d'),
            'harvest_date': harvest_date.strftime('%Y-%m-%d'),
            'actual_yield_tons_per_hectare': actual_yield,
            'variety_id': variety['id'],
            'variety_name': variety['name'],
            'maturity_days': variety['maturity_days'],
            'drought_resistance': variety['drought_resistance'],
            'optimal_temperature_min': variety['optimal_temp_min'],
            'optimal_temperature_max': variety['optimal_temp_max'],
            'disease_resistance': variety['disease_resistance'],
            'agro_ecological_zone': agro_zone,
            'location': location,
            'soil_data': soil_data,
            'weather_data': weather_data,
            'seed_rate_kg_per_hectare': round(management['seed_rate_kg_per_hectare'], 1),
            'row_spacing_cm': management['row_spacing_cm'],
            'fertilizer_type': management['fertilizer_type'],
            'fertilizer_amount_kg_per_hectare': round(management['fertilizer_amount_kg_per_hectare'], 1),
            'irrigation_method': management['irrigation_method'],
            'notes': f"Synthetic data for {agro_zone} zone, {season['name']} planting"
        }

        return sample

    def generate_dataset(self, num_samples: int, output_path: str,
                         years: List[int] = None) -> None:
        """Generate complete synthetic dataset"""

        if years is None:
            years = [2020, 2021, 2022, 2023]

        print(f"Generating {num_samples} synthetic training samples...")

        samples = []
        samples_per_year = num_samples // len(years)

        for year in years:
            year_samples = []
            for i in range(samples_per_year):
                sample_id = len(samples) + i + 1
                try:
                    sample = self.generate_sample(sample_id, year)
                    year_samples.append(sample)
                except Exception as e:
                    print(f"Error generating sample {sample_id}: {e}")
                    continue

            samples.extend(year_samples)
            print(f"Generated {len(year_samples)} samples for year {year}")

        # Generate remaining samples for the last year
        remaining = num_samples - len(samples)
        for i in range(remaining):
            sample_id = len(samples) + i + 1
            try:
                sample = self.generate_sample(sample_id, years[-1])
                samples.append(sample)
            except Exception as e:
                print(f"Error generating sample {sample_id}: {e}")
                continue

        # Save dataset
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(samples, f, indent=2, default=str)

        # Generate summary statistics
        self.generate_summary(samples, output_path.parent / 'dataset_summary.json')

        print(f"\nDataset generated successfully!")
        print(f"Total samples: {len(samples)}")
        print(f"Output file: {output_path}")
        print(f"Summary file: {output_path.parent / 'dataset_summary.json'}")

    def generate_summary(self, samples: List[Dict], summary_path: Path) -> None:
        """Generate dataset summary statistics"""

        yields = [s['actual_yield_tons_per_hectare'] for s in samples]
        varieties = [s['variety_name'] for s in samples]
        zones = [s['agro_ecological_zone'] for s in samples]

        summary = {
            'total_samples': len(samples),
            'yield_statistics': {
                'mean': round(np.mean(yields), 2),
                'std': round(np.std(yields), 2),
                'min': round(np.min(yields), 2),
                'max': round(np.max(yields), 2),
                'median': round(np.median(yields), 2),
                'percentiles': {
                    '25th': round(np.percentile(yields, 25), 2),
                    '75th': round(np.percentile(yields, 75), 2),
                    '90th': round(np.percentile(yields, 90), 2)
                }
            },
            'variety_distribution': {variety: varieties.count(variety) for variety in set(varieties)},
            'zone_distribution': {zone: zones.count(zone) for zone in set(zones)},
            'temporal_distribution': {},
            'data_quality': {
                'samples_with_weather': sum(1 for s in samples if s['weather_data']),
                'samples_with_soil': sum(1 for s in samples if s['soil_data']),
                'complete_samples': sum(1 for s in samples if s['weather_data'] and s['soil_data'])
            }
        }

        # Temporal distribution
        years = [s['planting_date'][:4] for s in samples]
        summary['temporal_distribution'] = {year: years.count(year) for year in set(years)}

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nDataset Summary:")
        print(f"Mean yield: {summary['yield_statistics']['mean']} tons/ha")
        print(f"Yield range: {summary['yield_statistics']['min']} - {summary['yield_statistics']['max']} tons/ha")
        print(f"Variety distribution: {summary['variety_distribution']}")
        print(f"Zone distribution: {summary['zone_distribution']}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic training data for maize yield prediction')
    parser.add_argument('--samples', type=int, default=2000, help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='./data/training/synthetic_training_data.json',
                        help='Output file path')
    parser.add_argument('--years', nargs='+', type=int, default=[2020, 2021, 2022, 2023],
                        help='Years to generate data for')

    args = parser.parse_args()

    generator = SyntheticDataGenerator()
    generator.generate_dataset(args.samples, args.output, args.years)


if __name__ == "__main__":
    main()