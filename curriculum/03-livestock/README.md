# Module 3: Livestock Management & AI

**Duration:** 3 weeks  
**Prerequisites:** Modules 1-2  
**Difficulty:** Intermediate

---

## Module Overview

Apply computer vision and AI to livestock monitoring, health assessment, and productivity optimization. Students build systems that track animal behavior, detect health issues, and optimize feed efficiency.

**Students learn:**
- Computer vision for animal identification and behavior analysis
- Wearable sensors and IoT for livestock monitoring
- Machine learning for health prediction and early disease detection
- Automated feeding systems and nutrition optimization
- Economic analysis of livestock technology investments

**Real Value:** Livestock technology specialists earn $48-72k annually, with smart livestock management systems showing 15-25% improvement in productivity and 20-30% reduction in veterinary costs.

---

## Week 1: Computer Vision for Animal Monitoring

### Hands-On Project: Cattle Identification System
```python
import cv2
import numpy as np
from tensorflow import keras
import datetime

class CattleMonitoringSystem:
    def __init__(self):
        self.animal_database = {}
        self.behavior_classifier = self.load_behavior_model()
        
    def identify_individual_animal(self, image):
        """Use computer vision to identify individual cattle"""
        # Process image for unique features (coat patterns, size, markings)
        features = self.extract_animal_features(image)
        
        # Match against database
        animal_id = self.match_to_database(features)
        
        return {
            'animal_id': animal_id,
            'confidence': 0.89,
            'timestamp': datetime.datetime.now().isoformat(),
            'location': 'North Pasture - Camera 3'
        }
    
    def analyze_behavior(self, video_sequence):
        """Analyze animal behavior patterns from video"""
        behaviors = {
            'grazing': 45,  # percentage of time
            'resting': 35,
            'walking': 15,
            'drinking': 3,
            'social_interaction': 2
        }
        
        # Detect anomalies
        alerts = []
        if behaviors['grazing'] < 30:
            alerts.append("LOW GRAZING: Possible health issue or feed quality problem")
        if behaviors['social_interaction'] < 1:
            alerts.append("ISOLATION: Animal may be sick or stressed")
            
        return {
            'behavior_summary': behaviors,
            'health_alerts': alerts,
            'recommendation': self.get_health_recommendation(behaviors)
        }

    def get_health_recommendation(self, behaviors):
        """Generate health recommendations based on behavior analysis"""
        if behaviors['grazing'] < 30:
            return "Schedule veterinary examination within 24 hours"
        elif behaviors['resting'] > 60:
            return "Monitor for lameness or illness"
        else:
            return "Normal behavior patterns observed"
```

---

## Week 2: Precision Livestock Farming

### Hands-On Project: Smart Feeding System
```python
class SmartFeedingSystem:
    def __init__(self):
        self.feed_database = {
            'corn_silage': {'cost_per_ton': 45, 'dry_matter': 0.35, 'energy_mcal_lb': 0.70},
            'alfalfa_hay': {'cost_per_ton': 180, 'dry_matter': 0.89, 'energy_mcal_lb': 0.60},
            'soybean_meal': {'cost_per_ton': 420, 'dry_matter': 0.89, 'protein_percent': 48}
        }
        
    def optimize_ration(self, animal_weight, milk_production, feed_prices):
        """Optimize feed ration for individual animals"""
        
        # Calculate nutritional requirements
        energy_need = self.calculate_energy_requirement(animal_weight, milk_production)
        protein_need = self.calculate_protein_requirement(animal_weight, milk_production)
        
        # Optimize feed mix for minimum cost
        optimal_ration = self.linear_programming_optimization(
            energy_need, protein_need, feed_prices
        )
        
        return {
            'daily_feed_cost': round(optimal_ration['cost'], 2),
            'feed_amounts': optimal_ration['amounts'],
            'nutritional_adequacy': optimal_ration['adequacy'],
            'cost_savings_vs_standard': round(optimal_ration['savings'], 2)
        }
    
    def calculate_energy_requirement(self, weight, milk_production):
        """Calculate daily energy requirements (Mcal)"""
        maintenance = weight * 0.08  # Maintenance energy
        production = milk_production * 0.7  # Production energy
        return maintenance + production
    
    def monitor_feed_efficiency(self, intake_data, production_data):
        """Calculate and monitor feed conversion efficiency"""
        
        efficiency_metrics = {}
        
        for animal_id in intake_data:
            daily_intake = intake_data[animal_id]['dry_matter_lbs']
            daily_production = production_data[animal_id]['milk_lbs']
            
            feed_efficiency = daily_production / daily_intake if daily_intake > 0 else 0
            
            efficiency_metrics[animal_id] = {
                'feed_conversion_ratio': round(feed_efficiency, 2),
                'status': self.evaluate_efficiency(feed_efficiency),
                'improvement_potential': self.calculate_improvement_potential(feed_efficiency)
            }
        
        return efficiency_metrics
    
    def evaluate_efficiency(self, ratio):
        """Evaluate feed conversion efficiency"""
        if ratio > 1.8:
            return "EXCELLENT"
        elif ratio > 1.5:
            return "GOOD"
        elif ratio > 1.2:
            return "AVERAGE"
        else:
            return "POOR - Needs attention"
```

---

## Week 3: Health Monitoring and Disease Prevention

### Hands-On Project: Early Disease Detection System
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class LivestockHealthMonitor:
    def __init__(self):
        self.health_model = self.train_disease_prediction_model()
        self.normal_ranges = {
            'body_temperature': (100.5, 102.5),  # Fahrenheit
            'heart_rate': (60, 80),  # BPM
            'rumination_minutes': (420, 540),  # Minutes per day
            'activity_steps': (3000, 8000),  # Steps per day
            'milk_production_lbs': (50, 90)  # Pounds per day
        }
    
    def continuous_health_monitoring(self, sensor_data):
        """Process continuous sensor data for health assessment"""
        
        health_score = 100  # Start with perfect health score
        alerts = []
        
        # Check each vital sign against normal ranges
        for metric, value in sensor_data.items():
            if metric in self.normal_ranges:
                normal_min, normal_max = self.normal_ranges[metric]
                
                if value < normal_min:
                    health_score -= 15
                    alerts.append(f"LOW {metric.upper()}: {value} (normal: {normal_min}-{normal_max})")
                elif value > normal_max:
                    health_score -= 10
                    alerts.append(f"HIGH {metric.upper()}: {value} (normal: {normal_min}-{normal_max})")
        
        # Predict disease risk using ML model
        risk_prediction = self.predict_disease_risk(sensor_data)
        
        return {
            'health_score': max(0, health_score),
            'disease_risk': risk_prediction,
            'immediate_alerts': alerts,
            'recommendations': self.generate_health_recommendations(health_score, alerts)
        }
    
    def predict_disease_risk(self, current_data):
        """Use machine learning to predict disease probability"""
        
        # Feature engineering from sensor data
        features = [
            current_data.get('body_temperature', 101.5),
            current_data.get('heart_rate', 70),
            current_data.get('rumination_minutes', 480),
            current_data.get('activity_steps', 5000),
            current_data.get('milk_production_lbs', 70)
        ]
        
        # Predict using trained model (simplified)
        risk_score = np.random.uniform(0.1, 0.9)  # Replace with actual model prediction
        
        if risk_score > 0.7:
            risk_level = "HIGH"
        elif risk_score > 0.4:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        return {
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'confidence': 0.84
        }
    
    def generate_health_recommendations(self, health_score, alerts):
        """Generate actionable health recommendations"""
        
        if health_score < 60:
            return "URGENT: Contact veterinarian immediately"
        elif health_score < 80:
            return "Schedule veterinary examination within 24 hours"
        elif len(alerts) > 0:
            return "Continue monitoring closely, document any changes"
        else:
            return "Animal health appears normal, continue routine monitoring"

# Economic impact calculator
class LivestockROICalculator:
    def calculate_technology_roi(self, herd_size, technology_cost, annual_savings):
        """Calculate ROI for livestock monitoring technology"""
        
        # Typical benefits from precision livestock farming
        health_cost_reduction = herd_size * 150  # $150 per head annually
        feed_efficiency_savings = herd_size * 200  # $200 per head annually
        reproduction_improvement = herd_size * 100  # $100 per head annually
        
        total_annual_benefits = health_cost_reduction + feed_efficiency_savings + reproduction_improvement
        
        roi_years = technology_cost / total_annual_benefits if total_annual_benefits > 0 else 0
        
        return {
            'herd_size': herd_size,
            'technology_investment': technology_cost,
            'annual_benefits': total_annual_benefits,
            'payback_period_years': round(roi_years, 1),
            'five_year_roi_percent': round(((total_annual_benefits * 5 - technology_cost) / technology_cost) * 100, 1)
        }

# Demo usage
health_monitor = LivestockHealthMonitor()
roi_calculator = LivestockROICalculator()

# Example sensor data
sample_data = {
    'body_temperature': 101.8,
    'heart_rate': 75,
    'rumination_minutes': 450,
    'activity_steps': 4200,
    'milk_production_lbs': 65
}

health_assessment = health_monitor.continuous_health_monitoring(sample_data)
roi_analysis = roi_calculator.calculate_technology_roi(100, 50000, 35000)

print("=== LIVESTOCK HEALTH MONITORING ===")
print(f"Health Score: {health_assessment['health_score']}/100")
print(f"Disease Risk: {health_assessment['disease_risk']['risk_level']}")
print(f"Recommendation: {health_assessment['recommendations']}")

print("\n=== TECHNOLOGY ROI ANALYSIS ===")
print(f"Payback Period: {roi_analysis['payback_period_years']} years")
print(f"5-Year ROI: {roi_analysis['five_year_roi_percent']}%")
```

---

## Assessment

### Technical Projects (60%)
- Computer vision animal identification system
- Smart feeding optimization calculator  
- Health monitoring and alert system
- ROI analysis for livestock technology implementation

### Industry Application (25%)
- Partner with local dairy or beef operation
- Implement monitoring system on real animals
- Document measurable improvements in efficiency or health

### Professional Presentation (15%)
- Present livestock technology solution to farmers and veterinarians
- Demonstrate cost-benefit analysis and practical implementation
- Address questions about animal welfare and technology adoption

---

## Career Connections

**Entry to Mid-Level Positions:**
- Livestock Technology Specialist: $48-68k
- Dairy Systems Analyst: $52-72k
- Animal Health Technology Coordinator: $45-62k
- Precision Livestock Consultant: $58-85k

**Industry Partners:**
- Dairy equipment manufacturers (DeLaval, GEA, Lely)
- Animal health companies (Zoetis, Merck Animal Health)
- Livestock monitoring technology (Allflex, SCR, CowManager)
- Agricultural cooperatives and large dairy operations

---

*Module Complete - Prepares Students for Module 4: Drones & Autonomous Systems*