import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import unittest

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    if img.ndim == 2:
        img = np.stack((img,) * 3, axis=-1)
    img = img.astype("float32")
    img = preprocess_input(img)
    return img

class TestPreprocessing(unittest.TestCase):
    def test_preprocess_image_output_shape(self):
        dummy_img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        processed = preprocess_image(dummy_img)
        self.assertEqual(processed.shape, (224, 224, 3))

    def test_preprocess_image_value_range(self):
        dummy_img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        processed = preprocess_image(dummy_img)
        self.assertTrue(np.max(processed) <= 1.0 and np.min(processed) >= -1.0)

    def test_preprocess_grayscale_conversion(self):
        dummy_gray = np.random.randint(0, 256, (300, 300), dtype=np.uint8)
        processed = preprocess_image(dummy_gray)
        self.assertEqual(processed.shape, (224, 224, 3))

if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)
