import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from PIL import Image
import cv2

class ModelHandler:
    def __init__(self):
        self.cnn_model = None
        self.ml_model = None
        self.word_model = None
        self.load_models()

    def load_models(self):
        try:
            self.cnn_model = load_model('models/cnn_az_handwritten_model.keras', compile=False)
            self.ml_model = joblib.load('models/random_forest_az_handwritten.joblib')
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
    
    def preprocess_image_for_ml(self, image_path):
        try:
            img = Image.open(image_path).convert('L')
            img = img.resize((28, 28))
            img_array = np.array(img)
            img_array = 255 - img_array
            img_array = img_array / 255.0
            img_flattened = img_array.reshape(1, -1)
            
            return img_flattened
        except Exception as e:
            print(f"Preprocessing error: {str(e)}")
            return None
    
    def preprocess_image_for_cnn(self, image_path):
        try:
            img = Image.open(image_path).convert('L')
            img = img.resize((28, 28))
            img_array = np.array(img)
            img_array = 255 - img_array
            img_array = img_array / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)
            return img_array
        
        except Exception as e:
            print(f"Preprocessing error: {str(e)}")
            return None
        
    def ml_predict(self, image_path):
        """Predict using traditional ML model"""
        # Preprocess
        img = self.preprocess_image_for_ml(image_path)
        if img is None:
            return "Error in preprocessing"

        # Predict
        prediction = self.ml_model.predict(img)[0]
        decoded = self.decode_prediction(prediction)
        return decoded
    
    def cnn_predict(self, image_path):
        """Predict using CNN model"""
        img = self.preprocess_image_for_cnn(image_path)
        if img is None:
            return "Error in preprocessing"
        
        prediction = self.cnn_model.predict(img)
        predicted_class = np.argmax(prediction)
        decoded = self.decode_prediction(predicted_class)
        return decoded
    
    
    def decode_prediction(self, pred):
        """Decode output to text"""
        word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
        predicted_letter = word_dict.get(pred, "Unknown")
        
        return predicted_letter
    
    
if __name__ == "__main__":
    handler = ModelHandler()
    test_image_path = "is.png"

    # print("CNN Prediction:")
    # cnn_result = handler.cnn_predict(test_image_path)
    # print(f"Predicted Letter (CNN): {cnn_result}")

    # print("ML Prediction:")
    # ml_result = handler.ml_predict(test_image_path)
    # print(f"Predicted Letter (ML): {ml_result}")

