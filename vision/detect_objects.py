from ultralytics import YOLO

# Note: The default YOLOv8 model is not specifically trained niche objects. List of the recognized classes here https://github.com/ultralytics/yolov5/blob/master/data/coco.yaml
# The model is loaded once globally to avoid redundant loading during each function call.
model = YOLO("yolov8n.pt")

def detect_objects(image_path: str) -> list[tuple[int, int, int, int]]:
    """
    Detects objects in an image using the YOLOv8 model and returns a list of bounding boxes.
    
    Args:
        image_path (str): Path to the input image.
    
    Returns:
        List[Tuple[int, int, int, int]]: A list of bounding box coordinates in (x1, y1, x2, y2) format.
    """
    # Run object detection on the input image (on CPU)
    results = model(image_path, device='cpu')

    bboxes = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Convert box coordinates to integers and extract (x1, y1, x2, y2)
            coords = box.xyxy.cpu().numpy().astype(int)[0]
            bboxes.append(tuple(coords))
    return bboxes
