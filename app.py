from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import numpy as np
import onnxruntime as ort
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "waste_classifier.onnx")

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# Load model once
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image = np.array(image).astype(np.float32)

    # If your CNN was trained with /255 normalization
    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    return image


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    try:
        image = Image.open(file.stream)

        processed = preprocess_image(image)

        output = session.run(
            None,
            {input_name: processed}
        )

        predictions = output[0][0]

        class_index = int(np.argmax(predictions))

        confidence = float(predictions[class_index])

        # If model output is logits, convert to probabilities
        if confidence > 1:
            exp_values = np.exp(
                predictions - np.max(predictions)
            )
            probabilities = exp_values / np.sum(exp_values)
            confidence = float(probabilities[class_index])

        prediction = CLASS_NAMES[class_index]

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )