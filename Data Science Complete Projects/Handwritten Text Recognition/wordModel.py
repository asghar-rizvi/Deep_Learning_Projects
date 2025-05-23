import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import StringLookup
import json

class HandwritingRecognizer:
    def __init__(self, model_path='models/handwriting_prediction_model.keras'):
        self.model = keras.models.load_model(model_path, compile=False)
        self.img_width = 128 
        self.img_height = 32
        self.max_len = 17
        
        with open('models/mapping_word_model/char_mapping.json') as f:
            self.characters = json.load(f) 
        
        self.char_to_num = StringLookup(
            vocabulary=self.characters,
            mask_token=None
        )
        self.num_to_char = StringLookup(
            vocabulary=self.characters,
            mask_token=None,
            invert=True
        )
        
        
    def distortion_free_resize(self, image, img_size):
        w, h = img_size
        image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)
        pad_height = h - tf.shape(image)[0]
        pad_width = w - tf.shape(image)[1]
        if pad_height % 2 != 0:
            height = pad_height // 2
            pad_height_top = height + 1
            pad_height_bottom = height
        else:
            pad_height_top = pad_height_bottom = pad_height // 2

        if pad_width % 2 != 0:
            width = pad_width // 2
            pad_width_left = width + 1
            pad_width_right = width
        else:
            pad_width_left = pad_width_right = pad_width // 2

        image = tf.pad(
            image,
            paddings=[
                [pad_height_top, pad_height_bottom],
                [pad_width_left, pad_width_right],
                [0, 0],
            ],
        )

        image = tf.transpose(image, perm=[1,0,2])
        image = tf.image.flip_left_right(image)
        
        return image  
        
    def preprocess_image(self, image_path):
        img_size = (self.img_width, self.img_height)
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=1) 
        image = self.distortion_free_resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.expand_dims(image, axis=0) 
        image = tf.expand_dims(image, axis=-1) if len(image.shape) == 3 else image  
    
        return image

    def decode_predictions(self,pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
            :, :self.max_len
        ]
        
        output_text = []
        for res in results:
            res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
            res = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    def predict(self, image_path):
        processed_img = self.preprocess_image(image_path)
        preds = self.model.predict(processed_img)

        top_indices = np.argmax(preds, axis=-1)[0]
        decoded = []
        for i, idx in enumerate(top_indices):
            char = self.num_to_char(idx).numpy().decode('utf-8')
            confidence = preds[0, i, idx]
            decoded.append((char, float(confidence)))
            if i > 20:  
                break
    
        return self.decode_predictions(preds)[0]

if __name__ == "__main__":
    recognizer = HandwritingRecognizer()
    test_image_path = "on.png"
    
    print("Predicted text:", recognizer.predict(test_image_path))