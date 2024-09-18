import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 model
model_path = 'yolov8n.pt'
model = YOLO(model_path)

# Load a single image
image_path = 'path/to/your/image.jpg'  # Provide the path to your image file
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image.")
    exit()

# Perform inference
results = model(image)

# Process results
for result in results:
    boxes = result.boxes.xyxy.numpy()  # Bounding boxes
    scores = result.boxes.conf.numpy()  # Confidence scores
    labels = result.names  # Class labels

    # Draw bounding boxes on the image
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = map(int, box)
        label = f"{labels[int(result.boxes.cls[i])]} {score:.2f}"
        color = (0, 255, 0)  # Green for bounding boxes
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the image with detections
cv2.imshow('YOLOv8 Detection', image)
cv2.waitKey(0)  # Wait indefinitely until a key is pressed

# Release resources
cv2.destroyAllWindows()
