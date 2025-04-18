import cv2
import numpy as np
import time
import hand_tracker_module as htm
import mediapipe as mp
import os

# 
brushThickness = 15
eraserThickness = 100
# 
floder_path = r"C:\Users\Asus\Documents\COMPUTER VISION\aiImage"
myList = os.listdir(floder_path)
# print(myList)

overlayList = []
for imgPath in myList:
    image = cv2.imread(os.path.join(floder_path, imgPath))
    overlayList.append(image)
# print(len(overlayList))

header = overlayList[0]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = htm.handDetector(detectionCon=0.85)
drawColor = (0, 255, 0)  # Green color for drawing
xp, yp = 0, 0  # Previous x and y coordinates for drawing
imgCanvas = np.zeros((720, 1280, 3), np.uint8)  # Create a blank canvas for drawing


while True:
    # 1 Import Image
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip the image horizontally
    if not success:
        break

    # 2 Find Hand Landmarks
    img = detector.findHands(img)
    lmList, _ = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList)

        # tip of index finger and middle finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

    # 3. check which finger is up
        finger = detector.fingerUp()
        # print(finger)
    # 4. if selection mode - two fingers up
        if finger[1] == 1 and finger[2] == 1:
            xp, yp = 0, 0  # Reset previous coordinates
            print("Selection Mode")
            
            if y1 < 111:  # Finger is in header zone
                if 0 < x1 < 270:       # First option (approx 1/4 of 1079)
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 270 < x1 < 540:   # Second option
                    header = overlayList[1]
                    drawColor = (0, 0, 255)
                elif 540 < x1 < 810:   # Third option
                    header = overlayList[2]
                    drawColor = (0, 255, 255)
                elif 810 < x1 < 1079:  # Fourth option
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
        # 5. if drawing mode - one finger up
        if finger[1] == 1 and finger[2] == 0:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")

            if xp == 0 and yp == 0:  # If this is the first point, set xp and yp to x1 and y1
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                # Eraser mode
                cv2.circle(img, (x1, y1), eraserThickness, drawColor, cv2.FILLED)
                cv2.circle(imgCanvas, (x1, y1), eraserThickness, drawColor, cv2.FILLED)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1  # Update previous coordinates
    # imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)  # Convert canvas to grayscale
    # _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)  # Invert the grayscale image
    # imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)  # Convert back to BGR
    # imgInv = cv2.bitwise_and(img, imgInv)  # Apply the inverted mask to the original image
    # img = cv2.bitwise_or(img, imgCanvas)  # Apply the canvas to the original image
    

    
                    
#    # 6. Merge the image with the canvas
    img[0:111, 0:1079] = header
    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)  # Blend the canvas with the image
    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)
    # cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)