from tensorflow.keras.models import load_model
model = load_model('artifacts/training/model.h5')
model.summary()  # Check the number of units in the last Dense layer and its activationpyt