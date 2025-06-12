import pandas as pd
from vision.detect_objects import detect_objects
from vision.recognize_text import ocr_text_on_objects
from vision.describe_image import get_caption
from llm.llm_integration_api_rest import send_description_to_llm
from vision.utils import verification_items_from_llm_response, parse_llm_response_and_verify

if __name__ == "__main__":
    image_path = "assets/production_image_example.jpg"
    
    # Step 1: Get a semantic caption for the image using BLIP
    caption = get_caption(image_path)
    print("📸 Automatic description:", caption)
    
    # Step 2: Detect objects
    bboxes = detect_objects(image_path)

    # Step 3: Extract text using OCR for each bounding box (if needed)
    all_recognized_texts = []
    for bbox in bboxes:
        recognized_text = ocr_text_on_objects(image_path, bbox)
        print(f"🔤 Recognized text in bbox {bbox}: {recognized_text}")
        all_recognized_texts.append(recognized_text)

    # Step 4: Compose a prompt with all recognized text and the caption for LLM processing - Here a simple example with car
    combined_text = "\n".join(all_recognized_texts)
    prompt = (
        f"Image description:\n{caption}\n\n"
        f"Extracted texts from image regions:\n{combined_text}\n\n"
        "Based on the visual description and extracted text, identify any car brands (e.g., BMW, Audi, Toyota, etc.) that appear in the image. "
        "List the detected brands. If none are clearly identifiable, respond with 'No car brand detected.'"
    )

    # Step 5: Optional Send prompt to LLM and get the response
    llm_response = send_description_to_llm(prompt)
    print("🤖 LLM response:", llm_response)

    # Optional: parse and verify LLM response for further processing
    verified_items = parse_llm_response_and_verify(llm_response)
    print("✅ Verified items from LLM response:", verified_items)
