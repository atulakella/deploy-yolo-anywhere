import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 model
model_path = 'yolov8n.pt'
model = YOLO(model_path)

# Open video capture (0 for default webcam, or provide video file path)
cap = cv2.VideoCapture(0)  # Use video file path if not using webcam

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)

    # Process results
    for result in results:
        boxes = result.boxes.xyxy.numpy()  # Bounding boxes
        scores = result.boxes.conf.numpy()  # Confidence scores
        labels = result.names  # Class labels

        # Draw bounding boxes on the frame
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = map(int, box)
            label = f"{labels[int(result.boxes.cls[i])]} {score:.2f}"
            color = (0, 255, 0)  # Green for bounding boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame with detections
    cv2.imshow('YOLOv8 Detection', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
