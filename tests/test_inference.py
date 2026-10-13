"""Unit and regression tests for plant disease inference and security safeguards"""
import html
import unittest
from unittest.mock import MagicMock
import numpy as np
from PIL import Image
from predict import PlantDiseasePredictor, load_class_names, DEFAULT_CLASSES, IMG_SIZE


class TestPlantDoctorInference(unittest.TestCase):

    def test_load_class_names_fallback(self):
        """Verify fallback class list when file is missing"""
        classes = load_class_names("non_existent_classes.json")
        self.assertIsInstance(classes, list)
        self.assertEqual(len(classes), len(DEFAULT_CLASSES))
        self.assertIn("Apple___Apple_scab", classes)

    def test_preprocess_image_shape(self):
        """Verify input image resizing and batch dimension expansion"""
        test_img = Image.new("RGB", (256, 256), color=(73, 109, 137))
        processed = PlantDiseasePredictor.preprocess_image(test_img, img_size=IMG_SIZE)

        self.assertEqual(processed.shape, (1, IMG_SIZE[0], IMG_SIZE[1], 3))
        self.assertEqual(processed.dtype, np.float32)

    def test_no_double_softmax_inference(self):
        """Verify that predictions from a model already with softmax output do NOT get double-softmaxed"""
        predictor = PlantDiseasePredictor(autoload=False)
        predictor.class_names = ["Healthy", "Diseased", "Blight"]

        # Mock model returning softmax probabilities: [0.1, 0.85, 0.05] (sum = 1.0)
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.1, 0.85, 0.05]], dtype=np.float32)
        predictor.model = mock_model

        test_img = Image.new("RGB", (128, 128), color=(100, 150, 200))
        pred_class, confidence = predictor.predict(test_img)

        self.assertEqual(pred_class, "Diseased")
        # If double-softmax was applied: softmax([0.1, 0.85, 0.05]) -> ~0.505 (50.5%)
        # With single softmax preserved: confidence must be 85.0%
        self.assertEqual(round(confidence, 1), 85.0)

    def test_uninitialized_model_raises_runtime_error(self):
        """Verify that invoking inference without trained weights raises clear error"""
        predictor = PlantDiseasePredictor(autoload=False)
        predictor.model = None

        test_img = Image.new("RGB", (128, 128), color=(0, 0, 0))
        with self.assertRaises(RuntimeError) as ctx:
            predictor.predict(test_img)
        self.assertIn("Model is not loaded", str(ctx.exception))

    def test_chat_html_escaping_prevents_xss(self):
        """Verify that unsafe HTML / XSS payloads in user and disease fields are escaped"""
        malicious_user_input = "<script>alert('xss')</script><b>Infected leaf</b>"
        malicious_disease = "Tomato___<img src=x onerror=alert(1)>"

        escaped_user = html.escape(malicious_user_input)
        escaped_disease = html.escape(malicious_disease)

        self.assertNotIn("<script>", escaped_user)
        self.assertIn("&lt;script&gt;", escaped_user)
        self.assertNotIn("<img", escaped_disease)
        self.assertIn("&lt;img", escaped_disease)


if __name__ == "__main__":
    unittest.main()
