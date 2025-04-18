import cv2
import mediapipe as mp
import time
import face_meshModule as fmm

cam = cv2.VideoCapture(r"C:\Users\Asus\Documents\HAND_TRACKING\videos.mp4\6.mp4")
pTime = 0
detector = fmm.FaceMeshDetector()
while True:
    success, img = cam.read()
    if not success:
        break
    img = cv2.resize(img, (840,780))
    img = detector.find_face_mesh(img)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 8)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break