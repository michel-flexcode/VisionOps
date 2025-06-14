"""
📸 Test Module – Image Captioning with CLIP/BLIP

This test validates that the image captioning system correctly identifies car-related content.
It focuses on detecting either known car brands or general car-related keywords.

🧪 Note: These tests serve just as examples for a car-related use case.
They are not required to run the core app, but they demonstrate how to validate the behavior
of the image captioning pipeline — especially useful since CLIP/BLIP outputs can be inconsistent.

💡 To run from the project root:
    pytest tests/test_recognize_text.py
"""

import pytest
from vision.describe_image import get_caption

valid_car_brands = {
    "toyota", "bmw", "audi", "mercedes", "tesla",
    "volkswagen", "honda", "ford", "chevrolet"
}

car_keywords = {
    "car", "vehicle", "automobile", "sedan", "suv",
    "convertible", "pickup", "truck"
}

def test_caption_mentions_car_or_brand():
    caption = get_caption(test_image_path).lower()

    found_brand = any(brand in caption for brand in valid_car_brands)
    found_car_word = any(word in caption for word in car_keywords)

    assert found_brand or found_car_word, f'Caption did not mention a car or a valid brand. Caption: "{caption}"'

# Image test with a Mercedes - Should detect a Mercedes
test_image_path = "tests/assets/car_image_test.jpg"

def test_caption_mentions_mercedes():
    caption = get_caption(test_image_path)

    assert "mercedes" in caption.lower(), f'Expected "Mercedes" in caption, but got: "{caption}"'