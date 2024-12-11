import cv2

# RTSP URL (replace with your actual URL)
rtsp_url = "rtsp://desarrollo:Password123.@192.168.6.31:554/Streaming/Channels/302"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

# Create a CLAHE object (Optional: you can tune these parameters)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Apply CLAHE (contrast enhancement)
    enhanced_frame = cv2.split(frame)
    enhanced_frame = [clahe.apply(channel) for channel in enhanced_frame]
    enhanced_frame = cv2.merge(enhanced_frame)

    # Show the enhanced frame
    cv2.imshow('Enhanced Stream', enhanced_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()