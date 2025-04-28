import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class multiclass_PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load model
        model = load_model(os.path.join("artifacts/training", "model.h5"))

        # Load and preprocess the image
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Get predictions
        probabilities = model.predict(test_image)  # Array of probabilities for each class
        result = np.argmax(probabilities, axis=1)  # Get the class index with the highest probability

        # Map class indices to their corresponding labels
        class_labels = {0: "glioma", 1: "meningioma", 2: "no-tumor", 3: "pituitary"}
        prediction = class_labels[result[0]]

        # Return the prediction
        return [{"image": prediction}]