from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from models_Handling import ModelHandler

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

model_handler = ModelHandler()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    model_type = request.form.get('model_type', 'cnn').lower()
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get prediction
            if model_type == 'ml':
                result = model_handler.ml_predict(filepath)
            elif model_type == 'word':
                result = model_handler.word_predict(filepath)
            else:  # Default to CNN
                result = model_handler.cnn_predict(filepath)
            
            # Clean up
            os.remove(filepath)
            
            return jsonify({
                'prediction': result,
                'model_used': model_type
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)