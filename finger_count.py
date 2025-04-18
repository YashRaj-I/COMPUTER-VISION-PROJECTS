import cv2
import mediapipe as mp
import time
import os
import hand_tracker_module as htm

WCam, HCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3, WCam)
cam.set(4, HCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.75)

folderPath = r"C:\Users\Asus\Documents\COMPUTER VISION\fingerImage"
MyList = os.listdir(folderPath)
# print(MyList)
overlayList = []
for imgPath in MyList:
    image = cv2.imread(f"{folderPath}/{imgPath}")
    print(f"{folderPath}/{imgPath}")
    # Resize overlay image (e.g., 100x100 pixels)
    image = cv2.resize(image, (200, 200))
    overlayList.append(image)

tipIds = [4, 8, 12, 16, 20]  # Index of finger tips in the hand landmarks

while True:
    success, img = cam.read()
    if not success:
        print("Failed to capture image")
        break
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList)
        finger =[]
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                finger.append(1)
        else:
                finger.append(0)
        # 4 fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                finger.append(1)
            else:
                finger.append(0)
        # print(finger)
        totalFingers = finger.count(1)
        print(totalFingers)

        h, w, c = overlayList[totalFingers-1].shape
        img[0:h, 0:w] = overlayList[totalFingers-1]          #img[0:h, 0:w] #// 2 + overlayList[0][0:h, 0:w] // 2
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 370), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)   


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)


