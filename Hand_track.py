import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0) # 0 for webcam, or provide a video file path

mpHands = mp.solutions.hands # Initialize MediaPipe Hands
hands = mpHands.Hands() # Hands object with default parameters
mpDraw = mp.solutions.drawing_utils # Drawing utilities for landmarks and connections

pTime = 0 # Previous time for FPS calculation
cTime = 0  # Current time for FPS calculation

while True: 
    success, img = cap.read() # Read a frame from the webcam
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert the image to RGB format for MediaPipe
    results = hands.process(imgRGB) # Process the image and detect hands

    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks: # If hands are detected
        for handLms in results.multi_hand_landmarks: # Iterate through detected hands
            for id, lm in enumerate(handLms.landmark): # Iterate through landmarks of each hand
                h, w, c = img.shape # Get the shape of the image
                cx, cy = int(lm.x * w), int(lm.y * h)   # Calculate the coordinates of the landmark
                if id == 8:  # Index finger tip
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)  # Draw a circle at the index finger tip

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  # Draw landmarks and connections
            
    

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3) # Display FPS on the image

    cv2.imshow("Image", img)    # Show the image in a window
    cv2.waitKey(1)  # Wait for 1 ms before moving to the next frame
    # Press 'q' to exit the loop and close the webcam feed