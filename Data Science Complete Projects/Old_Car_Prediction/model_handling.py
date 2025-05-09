import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

def load_models():
    print('Loading Models')
    return {
        'ann': load_model('models/ann_model.keras'),
        'xgb': joblib.load('models/xgb_model.joblib'),
    }

def preprocess_input(input_data):
    processed = pd.DataFrame(0, index=[0], columns=[
        'mileage',
        'age',
        'engine_Small (≤1L)',
        'engine_Medium (1-1.6L)',
        'engine_Large (1.6-2L)',
        'engine_Premium (2-3L)',
        'engine_Performance (>3L)',
        'transmission_encoded',
        'transmission_other',
        'fuel_encoded',
    ])
    
    # 1. Process mileage 
    processed['mileage'] = input_data['mileage']
    
    # 2. Calculate age
    current_year = 2023  
    processed['age'] = current_year - input_data['year']
    
    # 3. Engine size category 
    engine_col = f"engine_{input_data['engine_category']}"
    if engine_col in processed.columns:
        processed[engine_col] = 1
    
    # 4. Transmission encoding
    transmission_map = {
        'Manual': 0,
        'Semi-Auto': 1,
        'Automatic': 2,
        'Other': 3
    }
    transmission = input_data['transmission']
    processed['transmission_encoded'] = transmission_map.get(transmission, 3)
    processed['transmission_other'] = 1 if transmission == 'Other' else 0
    
    # 5. Fuel type encoding
    fuel_encoding = {
        'Diesel': 0,
        'Petrol': 1,
        'Hybrid': 2,
        'Electric': 3,
        'Other': 4
    }
    processed['fuel_encoded'] = fuel_encoding.get(input_data['fuelType'], 4)
    
    return processed

def predict_price(option, input_data, models):
    processed = preprocess_input(input_data)
        
    if option == 'neural_network':
        pred_log = models['ann'].predict([processed])[0][0]
    else:
        xgb_input = processed.to_numpy()
        pred_log = models['xgb'].predict(xgb_input)[0]
        
    return np.expm1(pred_log)  

if __name__ == '__main__':
    models = load_models()
    input_data = {
    'year': 2020,
    'mileage': 15000,
    'transmission': 'Automatic',
    'fuelType': 'Diesel',
    'engine_category': 'Large (1.6-2L)'  
    }
    
    print(predict_price('ml_model', input_data, models))