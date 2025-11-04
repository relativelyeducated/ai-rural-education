# Module 2: Precision Agriculture & Data Science

**Duration:** 4 weeks  
**Prerequisites:** Module 1  
**Difficulty:** Intermediate

---

## Module Overview

Deep dive into precision agriculture technologies, IoT sensors, and data-driven farming decisions. Students build complete monitoring systems and analyze real farm data.

**Students learn:**
- IoT sensor networks and data collection
- GPS and GIS technologies for precision farming
- Statistical analysis and machine learning for yield optimization
- Variable rate application technologies
- Economic analysis of precision agriculture investments

**Real Value:** Precision agriculture specialists earn $45-75k annually, with the global market growing 13% yearly to reach $23.5 billion by 2030.

---

## Week 1: IoT Sensors and Data Collection

### Learning Objectives
- Build IoT sensor networks for agricultural monitoring
- Collect and process real-time field data
- Understand precision agriculture hardware ecosystem

### Hands-On Project: Smart Field Monitoring System
```python
import time
import json
from datetime import datetime
import serial  # For Arduino communication

class FieldSensorSystem:
    def __init__(self, field_name, location):
        self.field_name = field_name
        self.location = location
        self.sensors = {
            'soil_moisture': {'pin': 'A0', 'calibration': (0, 1023, 0, 100)},
            'soil_temperature': {'pin': 'A1', 'sensor_type': 'DS18B20'},
            'ambient_temperature': {'pin': 'A2', 'sensor_type': 'DHT22'},
            'humidity': {'pin': 'A3', 'sensor_type': 'DHT22'}
        }
        
    def read_sensors(self):
        """Read all sensors and return formatted data"""
        sensor_data = {
            'timestamp': datetime.now().isoformat(),
            'field_name': self.field_name,
            'location': self.location,
            'soil_moisture_percent': self.read_soil_moisture(),
            'soil_temp_f': self.read_soil_temperature(),
            'air_temp_f': self.read_air_temperature(),
            'humidity_percent': self.read_humidity()
        }
        return sensor_data
    
    def read_soil_moisture(self):
        """Convert soil moisture sensor reading to percentage"""
        # Simulated reading - replace with actual sensor code
        raw_value = 512  # Arduino analog read
        # Convert to percentage (dry=1023, wet=300)
        moisture_percent = 100 - ((raw_value - 300) / (1023 - 300) * 100)
        return max(0, min(100, moisture_percent))
    
    def irrigation_recommendation(self, crop_type='corn'):
        """Generate irrigation recommendations based on sensor data"""
        data = self.read_sensors()
        
        thresholds = {
            'corn': {'optimal': (60, 80), 'critical': 40},
            'soybeans': {'optimal': (55, 75), 'critical': 35},
            'wheat': {'optimal': (50, 70), 'critical': 30}
        }
        
        moisture = data['soil_moisture_percent']
        crop_settings = thresholds.get(crop_type, thresholds['corn'])
        
        if moisture < crop_settings['critical']:
            return "URGENT: Immediate irrigation required"
        elif moisture < crop_settings['optimal'][0]:
            return "SCHEDULE: Irrigation needed within 24 hours"
        elif moisture > crop_settings['optimal'][1]:
            return "GOOD: Soil moisture optimal"
        else:
            return "MONITOR: Continue monitoring, no action needed"

# Usage example
field_monitor = FieldSensorSystem("North Field", {"lat": 40.7128, "lon": -74.0060})
current_data = field_monitor.read_sensors()
irrigation_advice = field_monitor.irrigation_recommendation('corn')

print(f"Field Data: {json.dumps(current_data, indent=2)}")
print(f"Recommendation: {irrigation_advice}")
```

---

## Week 2: GPS and Variable Rate Technology

### Learning Objectives
- Understand GPS accuracy and precision requirements
- Design variable rate application maps
- Calculate economic benefits of precision application

### Hands-On Project: Variable Rate Fertilizer Calculator
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

class VariableRateCalculator:
    def __init__(self, field_data):
        self.field_data = field_data  # GPS coordinates with soil test data
        
    def create_application_map(self, nutrient='nitrogen'):
        """Create variable rate application map based on soil tests"""
        
        # Extract coordinates and nutrient levels
        x_coords = [point['longitude'] for point in self.field_data]
        y_coords = [point['latitude'] for point in self.field_data]
        nutrient_levels = [point[f'{nutrient}_ppm'] for point in self.field_data]
        
        # Create interpolated grid
        xi = np.linspace(min(x_coords), max(x_coords), 100)
        yi = np.linspace(min(y_coords), max(y_coords), 100)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        
        # Interpolate nutrient levels across field
        zi_grid = griddata((x_coords, y_coords), nutrient_levels, 
                          (xi_grid, yi_grid), method='cubic')
        
        return xi_grid, yi_grid, zi_grid
    
    def calculate_fertilizer_rates(self, target_yield=180):
        """Calculate variable fertilizer rates based on soil tests and yield goals"""
        
        fertilizer_recommendations = []
        
        for point in self.field_data:
            current_n = point['nitrogen_ppm']
            soil_om = point['organic_matter_percent']
            
            # Nitrogen rate calculation (simplified)
            base_n_need = target_yield * 1.1  # 1.1 lbs N per bushel target
            soil_n_credit = current_n * 0.05  # Convert ppm to lbs/acre
            om_credit = soil_om * 20  # Organic matter N release
            
            recommended_n = max(0, base_n_need - soil_n_credit - om_credit)
            
            fertilizer_recommendations.append({
                'longitude': point['longitude'],
                'latitude': point['latitude'],
                'nitrogen_rate_lbs_acre': round(recommended_n, 1),
                'cost_per_acre': round(recommended_n * 0.45, 2)  # $0.45/lb N
            })
        
        return fertilizer_recommendations
    
    def economic_analysis(self, uniform_rate=150):
        """Compare variable rate vs uniform application economics"""
        
        vr_recommendations = self.calculate_fertilizer_rates()
        total_acres = len(self.field_data) * 2.5  # Assume 2.5 acres per sample point
        
        # Variable rate costs
        vr_total_n = sum(rec['nitrogen_rate_lbs_acre'] for rec in vr_recommendations)
        vr_avg_rate = vr_total_n / len(vr_recommendations)
        vr_total_cost = sum(rec['cost_per_acre'] for rec in vr_recommendations) * 2.5
        
        # Uniform rate costs
        uniform_total_cost = uniform_rate * 0.45 * total_acres
        
        # Savings calculation
        cost_savings = uniform_total_cost - vr_total_cost
        
        return {
            'total_acres': total_acres,
            'variable_rate_avg': round(vr_avg_rate, 1),
            'uniform_rate': uniform_rate,
            'vr_total_cost': round(vr_total_cost, 2),
            'uniform_total_cost': round(uniform_total_cost, 2),
            'annual_savings': round(cost_savings, 2),
            'savings_per_acre': round(cost_savings / total_acres, 2)
        }

# Example usage with sample data
sample_field_data = [
    {'longitude': -93.2650, 'latitude': 44.9778, 'nitrogen_ppm': 12, 'organic_matter_percent': 3.2},
    {'longitude': -93.2648, 'latitude': 44.9780, 'nitrogen_ppm': 18, 'organic_matter_percent': 2.8},
    {'longitude': -93.2652, 'latitude': 44.9782, 'nitrogen_ppm': 8, 'organic_matter_percent': 4.1},
    # Add more data points...
]

vr_calculator = VariableRateCalculator(sample_field_data)
economics = vr_calculator.economic_analysis()

print("=== VARIABLE RATE FERTILIZER ANALYSIS ===")
print(f"Total Field Size: {economics['total_acres']} acres")
print(f"Variable Rate Average: {economics['variable_rate_avg']} lbs N/acre")
print(f"Uniform Rate: {economics['uniform_rate']} lbs N/acre")
print(f"Annual Savings: ${economics['annual_savings']:,}")
print(f"Savings per Acre: ${economics['savings_per_acre']}")
```

---

## Week 3: Yield Mapping and Analysis

### Learning Objectives
- Process and analyze yield monitor data
- Identify yield-limiting factors
- Create management zones for precision agriculture

### Hands-On Project: Yield Data Analysis System
```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class YieldAnalyzer:
    def __init__(self, yield_data_file):
        self.yield_data = pd.read_csv(yield_data_file)
        self.clean_data()
        
    def clean_data(self):
        """Clean and filter yield monitor data"""
        # Remove outliers and invalid readings
        self.yield_data = self.yield_data[
            (self.yield_data['yield'] > 50) & 
            (self.yield_data['yield'] < 300) &
            (self.yield_data['speed'] > 2) &
            (self.yield_data['speed'] < 8)
        ]
        
    def create_management_zones(self, n_zones=3):
        """Create management zones based on yield patterns"""
        # Use yield, elevation, and soil properties for clustering
        features = self.yield_data[['yield', 'elevation', 'organic_matter']].fillna(0)
        
        kmeans = KMeans(n_clusters=n_zones, random_state=42)
        self.yield_data['management_zone'] = kmeans.fit_predict(features)
        
        # Analyze each zone
        zone_analysis = {}
        for zone in range(n_zones):
            zone_data = self.yield_data[self.yield_data['management_zone'] == zone]
            zone_analysis[f'Zone_{zone+1}'] = {
                'avg_yield': round(zone_data['yield'].mean(), 1),
                'yield_std': round(zone_data['yield'].std(), 1),
                'area_acres': len(zone_data) * 0.1,  # Assume 0.1 acres per point
                'avg_elevation': round(zone_data['elevation'].mean(), 1),
                'recommendations': self.get_zone_recommendations(zone_data)
            }
        
        return zone_analysis
    
    def get_zone_recommendations(self, zone_data):
        """Generate management recommendations for each zone"""
        avg_yield = zone_data['yield'].mean()
        
        if avg_yield > 180:
            return "HIGH YIELD: Maintain current practices, consider higher plant populations"
        elif avg_yield > 150:
            return "MODERATE YIELD: Optimize fertility, consider drainage improvements"
        else:
            return "LOW YIELD: Investigate soil constraints, consider soil amendments"
    
    def profitability_analysis(self, corn_price=4.50):
        """Calculate profitability by management zone"""
        if 'management_zone' not in self.yield_data.columns:
            self.create_management_zones()
        
        results = {}
        for zone in self.yield_data['management_zone'].unique():
            zone_data = self.yield_data[self.yield_data['management_zone'] == zone]
            
            avg_yield = zone_data['yield'].mean()
            revenue_per_acre = avg_yield * corn_price
            
            # Estimate variable costs by yield level
            base_cost = 450  # Base cost per acre
            if avg_yield > 180:
                variable_cost = base_cost + 50  # Higher input costs for high yield areas
            elif avg_yield > 150:
                variable_cost = base_cost
            else:
                variable_cost = base_cost - 25  # Lower input costs for poor areas
            
            profit_per_acre = revenue_per_acre - variable_cost
            
            results[f'Zone_{zone+1}'] = {
                'avg_yield': round(avg_yield, 1),
                'revenue_per_acre': round(revenue_per_acre, 2),
                'variable_cost': variable_cost,
                'profit_per_acre': round(profit_per_acre, 2)
            }
        
        return results

# Example implementation
print("=== YIELD ANALYSIS SYSTEM ===")
print("This system processes combine yield monitor data to:")
print("1. Identify yield patterns and limiting factors")
print("2. Create management zones for variable rate applications") 
print("3. Calculate profitability by field area")
print("4. Generate recommendations for next season")
```

---

## Week 4: Economic Analysis and ROI

### Learning Objectives
- Calculate return on investment for precision agriculture technologies
- Analyze cost-benefit scenarios
- Present economic justification to farm managers

### Capstone Project: Precision Agriculture Business Plan

Students create comprehensive analysis including:
- Technology investment requirements
- Annual operating costs
- Projected benefits and savings
- Risk assessment and sensitivity analysis
- Implementation timeline

---

## Assessment

### Project Portfolio (50%)
- IoT monitoring system design and implementation
- Variable rate application maps and economic analysis
- Yield data analysis and management zone creation
- Precision agriculture ROI business case

### Technical Skills (30%)
- Sensor data collection and processing
- GIS mapping and spatial analysis
- Statistical analysis and interpretation
- Economic modeling and forecasting

### Industry Presentation (20%)
- Present precision agriculture solution to industry panel
- Demonstrate measurable benefits and ROI
- Field questions about technical implementation
- Show understanding of practical farming constraints

---

## Career Connections

**Direct Career Paths:**
- Precision Agriculture Specialist: $55-75k
- Agricultural Technology Consultant: $60-85k
- Farm Data Analyst: $45-65k
- Precision Equipment Technician: $42-58k

**Companies Hiring:**
- John Deere, Climate Corporation, Trimble Agriculture
- Local equipment dealers and farm cooperatives
- Agricultural consulting firms
- Technology startups in agricultural space

---

*Module Complete - Prepares Students for Module 3: Livestock Management & AI*