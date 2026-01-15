# --- FINAL TRAFFIC SIGN DETECTION CODE ---
from ultralytics import YOLO
import cv2

# 1. Load your best model weights
model = YOLO('best.pt') 

# 2. Open the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

print("STATUS: Camera is starting... Press 'q' to stop.")

while True:
    success, img = cap.read()
    if not success:
        break

    # 3. Detect traffic signs in live video
    results = model(img, stream=True)

    for r in results:
        # Draw boxes and labels on the live frame
        img = r.plot()

    # 4. Show the live output window
    cv2.imshow('Traffic Sign Recognition', img)

    # Press 'q' key to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
