import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cam = cv2.VideoCapture(r"C:\Users\Asus\Documents\HAND_TRACKING\videos.mp4\9.mp4")
# cam = cv2.VideoCapture(0)  # Use 0 for webcam
pTime = 0

new_width = 1080  # Adjust as needed
new_height = 780  # Adjust as needed
while True:
    success, img = cam.read()
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
            # cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 15, (0, 255, 0), 2)

    # Resize frame
    img = cv2.resize(img, (new_width, new_height))

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 8)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break