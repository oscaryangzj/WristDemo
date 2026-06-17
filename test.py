import cv2
import time


cap = cv2.VideoCapture (0)
if not cap.isOpened ():
    print ("Unable to open camera")
    exit ()


prev_time = time.time ()

while True:
    ret, frame = cap.read ()
    if not ret:
        print ("Unable to read frame")
        break

    current_time = time.time ()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText( frame, f"Hello OpenCV", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText( frame, f"Hello OpenCV", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow ("Demo", frame)

    key = cv2.waitKey (1) & 0xFF
    if key == ord ("q"):
        break
    if key == ord("s"):
        cv2.imwrite("screenshot.jpg", frame)
        print("Screenshot saved")

cap.release ()
cv2.destroyAllWindows ()