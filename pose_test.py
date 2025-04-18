import cv2
import time
import mediapipe as mp
import pose_Module as pm
cam = cv2.VideoCapture(r"C:\Users\Asus\Documents\HAND_TRACKING\videos.mp4\2.mp4")
detector = pm.PoseDetector()
pTime = 0

new_width, new_height = 1080, 780  # Resize dimensions

while True:
    success, img = cam.read()
    if not success:
        break  # Exit if video ends or can't read frame

    img = detector.findPose(img)  # Detect pose
    lmList = detector.findPosition(img)
    if len(lmList) !=0:
        print(lmList)
    else:
        print("No landmarks detected")
    img = cv2.resize(img, (new_width, new_height))  # ✅ Resize inside loop

    # FPS Calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 8)
    cv2.imshow("Pose Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Quit when 'q' is pressed

cam.release()
cv2.destroyAllWindows()