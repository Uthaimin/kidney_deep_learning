import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def preprocess_input(self, img_path):
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = img / 255.0  # Normalize the image
            return img
        except Exception as e:
            print(f"Error loading and preprocessing image: {e}")
            return None

    def predict(self):
        try:
            # Load the model
            model = load_model(os.path.join("my_model","model.h5"))

            # Preprocess the image
            test_image = self.preprocess_input(self.filename)
            if test_image is None:
                return [{"image": "Error in image preprocessing"}]

            # Generate prediction probabilities
            probabilities = model.predict(test_image)
            print(probabilities)

            # Generate prediction
            result = np.argmax(probabilities, axis=1)
            print(result)

            # Interpret prediction
            class_labels = ['Normal', 'Tumor']  # Define your class labels
            prediction = class_labels[result[0]]
            
            return [{"image": prediction}]
        except Exception as e:
            print(f"Error in prediction: {e}")
            return [{"image": "Error in prediction"}]
