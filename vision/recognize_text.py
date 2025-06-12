import cv2
import pytesseract

def ocr_text_on_objects(image_path: str, bbox: tuple = None) -> str:
    """
    Extracts text from an image or a specified bounding box area using OCR.
    
    Args:
        image_path (str): Path to the input image file.
        bbox (tuple, optional): Bounding box coordinates (x1, y1, x2, y2) to crop the image region. 
                                If None, processes the entire image.
    
    Returns:
        str: Cleaned text extracted from the image region.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Crop image if bounding box provided
    if bbox:
        x1, y1, x2, y2 = bbox
        img = img[y1:y2, x1:x2]

    # Convert to grayscale (often improves OCR accuracy)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Optional: apply thresholding or other preprocessing here if needed
    # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    # Use 'thresh' instead of 'gray' if that gives better results

    # Run OCR on the processed image region (could be an other language)
    text = pytesseract.image_to_string(gray, lang='eng')

    # Clean the OCR output text
    text = text.strip().replace('\n', ' ').replace('\x0c', '')

    return text
