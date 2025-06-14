## This example test is tailored for a car-related use case ##
## It is provided as an illustration — you don’t need it to run the program, 
## but writing such tests is essential since CLIP/BLIP outputs are not always 100% accurate. ##

# To run this test, execute from the project root:
# pytest tests/test_recognize_text.py

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