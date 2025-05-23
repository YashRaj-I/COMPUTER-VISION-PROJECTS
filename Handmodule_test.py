import cv2
import mediapipe as mp
import time
import hand_tracker_module as htm
pTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()
    
while True: 
    success, img = cap.read()
    if not success:
        break

    img = detector.findHands(img, draw=False) # Set draw=False to avoid drawing on the image
    lmList = detector.findPosition(img, draw=False) # Set draw=False to avoid drawing on the image
    if len(lmList) != 0:
        # print(lmList[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()