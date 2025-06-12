from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

def get_caption(image_path: str) -> str:
    """
    Generates a semantic caption for an input image using the BLIP model.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        str: A natural language description of the image.
    """
    # Load and prepare the image
    image = Image.open(image_path).convert("RGB")
    
    # Load BLIP processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Use CPU explicitly (or switch to 'cuda' if available)
    device = torch.device("cpu")
    model.to(device)

    # Preprocess image and generate caption
    inputs = processor(images=image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=100)
    
    # Decode and return the caption
    return processor.decode(output[0], skip_special_tokens=True)



