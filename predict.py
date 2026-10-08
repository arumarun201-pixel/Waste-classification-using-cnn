import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

MODEL_PATH = "models/waste_classifier.keras"
CLASS_PATH = "models/class_names.json"
IMG_SIZE = 128

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

def predict_image(image_path):
    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array, verbose=0)
    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[0][predicted_index]) * 100

    print("\n==============================")
    print("      WASTE CLASSIFICATION")
    print("==============================")
    print("Predicted Class :", predicted_class)
    print("Confidence      :", f"{confidence:.2f}%")
    print("==============================\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py image.jpg")
    else:
        predict_image(sys.argv[1])
