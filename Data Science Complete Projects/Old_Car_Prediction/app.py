from flask import Flask, render_template, request, jsonify
from model_handling import load_models, predict_price
import os

app = Flask(__name__)

# Load models at startup
models = load_models()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         user_data = {
#             'year': int(request.form['year']),
#             'mileage': int(request.form['mileage']),
#             'transmission': request.form['transmission'],
#             'fuelType': request.form['fuelType'],
#             'engine_category': request.form['engine_category']
#         }
        
#         selected_model = request.form['model_type']
        
#         print('Inside app.py')
#         print(user_data)
#         print(selected_model)
        
#         prediction = predict_price(selected_model, user_data, models)
        
#         return jsonify({
#             'status': 'success',
#             'prediction': f"£{prediction:,.2f}",
#             'details': user_data
#         })
    
#     except Exception as e:
#         return jsonify({
#             'status': 'error',
#             'message': str(e)
#         }), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        required_fields = ['year', 'mileage', 'transmission', 'fuelType', 'engine_category', 'model_type']
        if not all(field in request.form for field in required_fields):
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields'
            }), 400

        user_data = {
            'year': int(request.form['year']),
            'mileage': int(request.form['mileage']),
            'transmission': request.form['transmission'],
            'fuelType': request.form['fuelType'],
            'engine_category': request.form['engine_category']
        }
        
        selected_model = request.form['model_type']
        print('model_type\n\n\n\n')
        print(selected_model)
        
        prediction = predict_price(selected_model, user_data, models)
        
        print('Prediction Output: ', prediction)
        
        return jsonify({
            'status': 'success',
            'prediction': f"£{prediction:,.2f}",
            'details': user_data
        })
    
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': f"Invalid input: {str(e)}"
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
        
        
if __name__ == '__main__':
    app.run(debug=True)