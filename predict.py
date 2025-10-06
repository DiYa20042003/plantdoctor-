import numpy as np
import json
import os
from PIL import Image
import streamlit as st

# ---------------- Config ----------------
IMG_SIZE = (128, 128)
MODEL_PATH = "models/plant_disease_model.keras"
CLASS_FILE = "models/class_names.json"

# ---------------- Load Model (Cached) ----------------
@st.cache_resource
def load_model():
    """
    Loads the trained model from disk. Returns None if the model doesn't exist.
    The st.cache_resource decorator ensures the model is loaded only once.
    """
    # TensorFlow is imported here to avoid slowing down the initial app launch,
    # as it's a heavy library and only needed for prediction.
    if os.path.exists(MODEL_PATH):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ Model loaded successfully.")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    print("⚠️ Model file not found at:", MODEL_PATH)
    return None

# ---------------- Load Class Names ----------------
try:
    with open(CLASS_FILE, "r") as f:
        class_names = json.load(f)
    print(f"✅ Loaded {len(class_names)} classes.")
except FileNotFoundError:
    print(f"❌ Class names file not found at: {CLASS_FILE}")
    class_names = [] # Default to an empty list if file is missing

# ---------------- Prediction Function ----------------
def predict_image(img_input):
    """
    Predicts the disease from an image.
    Handles model loading and potential errors gracefully.
    """
    model = load_model()
    if model is None:
        return "Model not found. Please train it first.", 0.0

    if not class_names:
        return "Class names not found. Check the JSON file.", 0.0

    # --- Load image depending on type ---
    try:
        if isinstance(img_input, str):  # file path
            img = Image.open(img_input).convert("RGB")
        elif hasattr(img_input, "read"):  # uploaded file-like object
            img = Image.open(img_input).convert("RGB")
        elif isinstance(img_input, Image.Image):  # already a PIL image
            img = img_input.convert("RGB")
        else:
            raise ValueError("Unsupported input type.")
    except Exception as e:
        return f"Error opening image: {e}", 0.0

    # --- Preprocess ---
    # TensorFlow is also imported here to access preprocessing functions.
    import tensorflow as tf
    img = img.resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img), axis=0)
    # The preprocessing must match what the model was trained with (EfficientNet).
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # --- Predict ---
    try:
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        predicted_class = class_names[np.argmax(score)]
        confidence = float(np.max(score) * 100)

        return predicted_class, confidence
    except Exception as e:
        return f"Prediction error: {e}", 0.0