import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image

# ---------------- Config ----------------
IMG_SIZE = (128, 128)
MODEL_PATH = "models/plant_disease_model.keras"
CLASS_FILE = "models/class_names.json"

# ---------------- Load Model ----------------
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_FILE, "r") as f:
    class_names = json.load(f)

print(f"✅ Loaded model with {len(class_names)} classes")

# ---------------- Prediction Function ----------------
def predict_image(img_input):
    """
    img_input can be:
    - str (file path)
    - BytesIO or UploadedFile (from browser uploader)
    - PIL Image
    """
    # --- Load image depending on type ---
    if isinstance(img_input, str):  # file path
        img = Image.open(img_input).convert("RGB")
    elif hasattr(img_input, "read"):  # uploaded file-like object
        img = Image.open(img_input).convert("RGB")
    elif isinstance(img_input, Image.Image):  # already a PIL image
        img = img_input.convert("RGB")
    else:
        raise ValueError("Unsupported input type. Provide file path, UploadedFile, or PIL image.")

    # --- Preprocess ---
    img = img.resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # --- Predict ---
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = float(np.max(score) * 100)

    return predicted_class, confidence
