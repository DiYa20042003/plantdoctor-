"""Plant disease inference pipeline using EfficientNetB0 transfer learning"""
import os
import json
from typing import Tuple, List, Optional, Union
from PIL import Image
import numpy as np

# ---------------- Config ----------------
IMG_SIZE = (128, 128)
DEFAULT_MODEL_PATH = os.path.join("models", "plant_disease_model.keras")
DEFAULT_CLASS_FILE = os.path.join("models", "class_names.json")

# Fallback class names for graceful development/testing if model not yet trained
DEFAULT_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___healthy",
    "Corn___Common_rust",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy"
]


def load_class_names(class_file_path: str = DEFAULT_CLASS_FILE) -> List[str]:
    """Load class names from JSON file or fall back to defaults"""
    if os.path.exists(class_file_path):
        try:
            with open(class_file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_CLASSES


class PlantDiseasePredictor:
    """Predictor encapsulating preprocessing, inference, and postprocessing"""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        class_file: str = DEFAULT_CLASS_FILE,
        autoload: bool = True
    ):
        self.model_path = model_path
        self.class_file = class_file
        self.class_names = load_class_names(class_file)
        self.model = None

        if autoload and os.path.exists(model_path):
            self.load_model()

    def load_model(self):
        """Load trained Keras model"""
        import tensorflow as tf
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
        return self.model

    @staticmethod
    def preprocess_image(img_input: Union[str, Image.Image, any], img_size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
        """
        Resize image and apply EfficientNet preprocessing.
        Accepts filepath, PIL Image, or file-like stream.
        """
        if isinstance(img_input, str):
            img = Image.open(img_input).convert("RGB")
        elif isinstance(img_input, Image.Image):
            img = img_input.convert("RGB")
        elif hasattr(img_input, "read"):
            img = Image.open(img_input).convert("RGB")
        else:
            raise ValueError("Unsupported image input type. Provide a path, PIL.Image, or file stream.")

        img = img.resize(img_size)
        arr = np.array(img, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)

        # EfficientNet expects [0, 255] RGB float inputs as handled by its preprocessor
        # Scaling is handled internally by EfficientNet's rescale layer or preprocess_input
        return arr

    def predict(self, img_input: Union[str, Image.Image, any]) -> Tuple[str, float]:
        """
        Run inference and return (predicted_class, confidence_percentage).
        Avoids double-softmax by checking output activation.
        """
        if self.model is None:
            raise RuntimeError(
                f"Model is not loaded. Train the model using train.py or place weights at {self.model_path}."
            )

        processed = self.preprocess_image(img_input)
        preds = self.model.predict(processed)
        probabilities = np.squeeze(preds)

        # If model outputs unnormalized logits (not summing to ~1.0), apply softmax
        prob_sum = np.sum(probabilities)
        if not np.isclose(prob_sum, 1.0, atol=1e-2):
            import tensorflow as tf
            probabilities = tf.nn.softmax(probabilities).numpy()

        predicted_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_idx] * 100.0)

        if predicted_idx < len(self.class_names):
            predicted_class = self.class_names[predicted_idx]
        else:
            predicted_class = f"Class_{predicted_idx}"

        return predicted_class, confidence


# Global module-level convenience function preserving backward compatibility
_default_predictor = None


def get_default_predictor() -> PlantDiseasePredictor:
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = PlantDiseasePredictor(autoload=True)
    return _default_predictor


def predict_image(img_input) -> Tuple[str, float]:
    """Compatibility wrapper for predict_image"""
    predictor = get_default_predictor()
    return predictor.predict(img_input)
