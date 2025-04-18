import cv2
import mediapipe as mp
import time
import numpy as np
import pose_Module as pm

count = 0
dir  = 0


# cam = cv2.VideoCapture(r"C:\Users\Asus\Documents\COMPUTER VISION\videos.mp4\7.mp4")
cam = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path
detector = pm.PoseDetector()
pTime = 0
cam.set(3, 1280)  # Set the width of the video capture
cam.set(4, 720)  # Set the height of the video capture
while True:
    success, image = cam.read()
    if not success:
        break

    # img = cv2.imread(r"C:\Users\Asus\Documents\COMPUTER VISION\videos.mp4\8.jpg")
    img = detector.findPose(image, draw=False)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)
    if len(lmList) != 0:
        angle = detector.findAngle(img, 11, 13, 15, draw=True) # Left Arm
        # angle =detector.findAngle(img, 12, 14, 16, draw=True) # Right Arm

        per = np.interp(angle, (210, 310), (0, 100))
        bar = np.interp(angle, (220, 310), (650, 100))
        # print(angle, per)

        # Check for the dumbbell curl
        color = (255, 0, 255)
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0
        # print(count)
        # Draw the progress bar
        cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED) 
        cv2.putText(img, f'{int(per)}%', (1100, 800), cv2.FONT_HERSHEY_PLAIN, 13, color, 5)


        cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 10)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (155, 200), cv2.FONT_HERSHEY_PLAIN, 13, (255, 0, 0), 5)

    image = cv2.resize(img, (1080, 780))  # Resize the image to the desired dimensions
    cv2.imshow("Image", image)
    cv2.waitKey(1)



