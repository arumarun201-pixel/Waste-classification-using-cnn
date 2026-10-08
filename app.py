import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
MODEL_PATH = "models/waste_classifier.keras"
CLASS_PATH = "models/class_names.json"
IMG_SIZE = 128

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

def predict_waste(image_path):
    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array, verbose=0)
    index = int(np.argmax(predictions[0]))
    predicted_class = CLASS_NAMES[index]
    confidence = float(predictions[0][index]) * 100
    return predicted_class, confidence

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", error="Please select an image.")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="Please select an image.")

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        predicted_class, confidence = predict_waste(file_path)
    except Exception as e:
        return render_template("index.html", error=f"Prediction error: {e}")

    return render_template(
        "index.html",
        prediction=predicted_class,
        confidence=f"{confidence:.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)
