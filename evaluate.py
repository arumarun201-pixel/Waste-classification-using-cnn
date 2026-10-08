import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_DIR = "dataset"
MODEL_PATH = "models/waste_classifier.keras"
IMG_SIZE = 128
BATCH_SIZE = 32

model = tf.keras.models.load_model(MODEL_PATH)

datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

test_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

loss, accuracy = model.evaluate(test_data)
print("\n==============================")
print("MODEL EVALUATION")
print("==============================")
print("Loss     :", round(loss, 4))
print("Accuracy :", round(accuracy * 100, 2), "%")

predictions = model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_data.classes
class_names = list(test_data.class_indices.keys())

print("\nClassification Report:\n")
print(classification_report(true_classes, predicted_classes, target_names=class_names))

cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(9, 7))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Waste Classification Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()
