import cv2
import datetime

# RTSP stream URL
rtsp_url = '<Your RRSP url here>'

# Initialize the video capture object
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
else:
    # Capture a single frame
    ret, frame = cap.read()

    if ret:
        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Add the timestamp to the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, timestamp, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Save the frame with the timestamp
        filename = f"frame_{timestamp}.jpg"
        cv2.imwrite(filename, frame)

        print(f"Frame saved as {filename}")
    else:
        print("Error: Could not read frame from RTSP stream.")

# Release the video capture object
cap.release()
