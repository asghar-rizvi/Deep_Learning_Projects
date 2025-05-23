import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import StringLookup
import json

class HandwritingRecognizer:
    def __init__(self, model_path='models/handwriting_prediction_model.keras'):
        """Initialize with the saved model"""
        # self.model = keras.models.load_model(model_path, compile=False)
        self.img_width = 128 
        self.img_height = 32
        self.max_len = 17
        
        with open('char_mapping.json') as f:
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

        # Check the amount of padding needed to be done.
        pad_height = h - tf.shape(image)[0]
        pad_width = w - tf.shape(image)[1]

        # Only necessary if you want to do same amount of padding on both sides.
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
        
        # 1. Read and decode (grayscale)
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=1)  # Explicit channels
        
        # 2. Resize with padding
        image = self.distortion_free_resize(image, img_size)
        print('\nImage Data\n', np.max(image))
        
        # 3. Normalize
        image = tf.cast(image, tf.float32) / 255.0
        
        print('\nImage Data After Division...: ', np.max(image))
        
        # 4. Add missing dimensions
        image = tf.expand_dims(image, axis=0)  # Add batch dimension
        image = tf.expand_dims(image, axis=-1) if len(image.shape) == 3 else image  # Ensure channel dim
        
        # Debug print
        print("\nFinal preprocessed shape:", image.shape)  # Should be (1, 128, 32, 1)
        
        return image

    def decode_predictions(self,pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        print('Input Length... :', input_len)
        
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
        print('\n\n', processed_img, '\n\n')
        preds = self.model.predict(processed_img)
        
        # Inspect raw predictions
        print("\nRaw prediction scores (first 5 timesteps):")
        print(preds[0, :5, :3])  # Show first 5 timesteps, top 3 characters
        
        # Get top predicted character indices
        top_indices = np.argmax(preds, axis=-1)[0]
        print("\nTop predicted indices:", top_indices)
        
        # Decode with confidence
        decoded = []
        for i, idx in enumerate(top_indices):
            char = self.num_to_char(idx).numpy().decode('utf-8')
            confidence = preds[0, i, idx]
            decoded.append((char, float(confidence)))
            if i > 20:  # Limit output
                break
        
        print("\nDecoded with confidence:")
        for char, conf in decoded:
            print(f"{char} ({conf:.2f})", end=' ')
        
        return self.decode_predictions(preds)[0]

if __name__ == "__main__":
    recognizer = HandwritingRecognizer()
    test_image_path = "the.png"
    
    print("Predicted text:", recognizer.preprocess_image(test_image_path))