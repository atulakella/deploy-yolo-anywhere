import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model_path = r"/home/aresuser/deploy-yolo-anywhere/best.pt"  # Path to your YOLOv8 model
model = YOLO(model_path)

# RTSP stream URL
rtsp_url = ''  # Update with your RTSP URL

# Initialize the video capture object
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

while True:
    # Capture a frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from RTSP stream.")
        break

    # Resize the frame for faster processing
    frame = cv2.resize(frame, (640, 480))  # Adjust the size as needed

    # Perform inference
    results = model(frame)

    if results is None:
        print("Error: No results returned from the model.")
        break

    # Process results
    for result in results:
        if result is None:
            continue

        if hasattr(result, 'boxes') and hasattr(result.boxes, 'xyxy'):
            boxes = result.boxes.xyxy.numpy()  # Bounding boxes
            scores = result.boxes.conf.numpy()  # Confidence scores
            labels = result.names  # Class labels

            # Draw bounding boxes and labels
            for i, (box, score) in enumerate(zip(boxes, scores)):
                x1, y1, x2, y2 = map(int, box)
                label = f"{labels[int(result.boxes.cls[i])]} {score:.2f}"
                color = (0, 255, 0)  # Green for bounding boxes
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            print("Error: Result does not have the expected attributes.")

    # Display the frame with detections
    cv2.imshow("Detections", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
