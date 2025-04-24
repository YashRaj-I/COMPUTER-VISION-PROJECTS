import numpy as np
import cv2
import time
import autopy
import hand_tracker_module as htm

Wcam, Hcam = 640, 480
frameR = 100  # Frame reduction ratio
smothening = 10  # Smoothening value for mouse movement
plocX, plocY = 0, 0  # Previous location of the mouse
clocX, clocY = 0, 0  # Current location of the mouse

clickCooldown = 0.8  # seconds
lastClickTime = 0


cam = cv2.VideoCapture(0)
cam.set(3, Wcam)
cam.set(4, Hcam)
pTime = 0
detector = htm.handDetector(maxHands=1)
Wscreen, Hscreen = autopy.screen.size()
# print(Wscreen, Hscreen)


while True:
    success, img = cam.read()
    if not success:
        break
    # 1. find hand landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # 2. Get the tip of the index and middle fingers
    if len(lmList)!=0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)

        # 3. Check which fingers are up
        fingers = detector.fingerUp()
        # print(fingers)

        cv2.rectangle(img, (frameR, frameR), (Wcam - frameR, Hcam - frameR), (255, 0, 255), 2)

        # # 4. Only index finger: Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert coordinates
            x3 = np.interp(x1, (frameR, Wcam - frameR), (0, Wscreen))
            y3 = np.interp(y1, (frameR, Hcam - frameR), (0, Hscreen))
            # 5. smoothen the values
            clocX = plocX + (x3 - plocX) / smothening
            clocY = plocY + (y3 - plocY) / smothening

            # 6. Move the mouse
            autopy.mouse.move(Wscreen - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.rectangle(img, (frameR, frameR), (Wcam - frameR, Hcam - frameR), (255, 0, 255), 2)
            plocX, plocY = clocX, clocY
        # # 7. Both index and middle fingers are up: Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 8. Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)
            # 9. Click mouse if distance is short
            # if length < 40:
            #     cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
            #     autopy.mouse.click()
            #     cv2.rectangle(img, (frameR, frameR), (Wcam - frameR, Hcam - frameR), (255, 0, 255), 2)

            import time  # already there
            if length < 40 and (time.time() - lastClickTime) > clickCooldown:
                lastClickTime = time.time()
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

# Display the image in a window
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
