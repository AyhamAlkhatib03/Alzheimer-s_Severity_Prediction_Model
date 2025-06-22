import cv2
import numpy as np
import pytest
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    if img.ndim == 2:
        img = np.stack((img,) * 3, axis=-1)
    img = img.astype("float32")
    img = preprocess_input(img)
    return img

@pytest.mark.parametrize("shape", [
    (300, 300, 3),   # RGB
    (300, 300)       # Grayscale
])
def test_preprocess_image_output_shape(shape):
    dummy_img = np.random.randint(0, 256, shape, dtype=np.uint8)
    processed = preprocess_image(dummy_img)
    assert processed.shape == (224, 224, 3)

def test_preprocess_image_value_range():
    dummy_img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    processed = preprocess_image(dummy_img)
    assert processed.max() <= 1.0 and processed.min() >= -1.0
