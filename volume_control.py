import cv2
import time
import mediapipe as mp
import numpy as np
import hand_tracker_module as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cam = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path
cam.set(3, 1280)  # Width
cam.set(4, 720)   # Height

pTime = 0
detector = htm.handDetector(detectionCon=0.75)

# Get default audio device
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()  # Get mute status 
# volume.GetMasterVolumeLevel()  # Get master volume level
volRange =(volume.GetVolumeRange())  # Get volume range

minVol = volRange[0]  # Minimum volume level
maxVol = volRange[1]  # Maximum volume level
vol = 0  # Initialize volume level
volBar = 400  # Initialize volume bar height
volPer = 0  # Initialize volume percentage





while True:
    suscess, img = cam.read()
    if not suscess:
        break  # Exit if video ends or can't read frame
    
    img = detector.findHands(img)  # Find hands in the image
    lmList = detector.findPosition(img, draw=False)  # Get landmark positions
    if len(lmList) != 0:
        # Example: Print the coordinates of the index finger tip
        # print(lmList[4], lmList[8])  # Index finger tip landmark is at index 8
        
        x1, y1 = lmList[4][1], lmList[4][2]  # Coordinates of the index finger tip
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2


        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)    # Draw a circle at the index finger tip
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)   # Draw a circle at the thumb tip
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)     # Draw a line between the index finger and thumb tips
        cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)   # Draw a circle at the center between the two tips

        length = math.hypot(x2 - x1, y2 - y1)  # Calculate the distance between the two tips
        # print(length)

        # Hand range 50 - 300
        # Volume range -65.25 to 0.0
        # vol = np.interp(length, [50, 300], [minVol, maxVol])
        vol = np.interp(length, [50, 300], [0.0, 1.0])  # Map length to volume scalar
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])

        print(int(length), vol)
        # Set volume to 50%
        volume.SetMasterVolumeLevelScalar(vol, None)


        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)  # Draw a rectangle for volume bar
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)  # Draw the volume bar
    cv2.putText(img, f'Volume: {int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 3)  # Display volume percentage

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 3)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)  # Wait for 1 ms to display the image