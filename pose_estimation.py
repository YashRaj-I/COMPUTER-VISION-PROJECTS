import cv2
import mediapipe as mp
import time

# Initialize MediaPipe pose and drawing utilities
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Load video file (or use webcam with cv2.VideoCapture(0))
cam = cv2.VideoCapture(r"videos.mp4/9.mp4")

# Time for FPS calculation
pTime = 0

# Desired frame size
new_width = 1080
new_height = 780

while True:
    success, img = cam.read()
    if not success:
        break  # Exit if video ends or frame not loaded

    # Convert BGR image to RGB for processing
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)

    # If pose landmarks are detected
    if results.pose_landmarks:
        # Draw landmarks and connections (red joints, green lines)
        mpDraw.draw_landmarks(
            img,
            results.pose_landmarks,
            mpPose.POSE_CONNECTIONS,
            mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),  # red joints
            mpDraw.DrawingSpec(color=(0, 255, 0), thickness=8)  # green lines
        )

        # Draw a magenta circle at each landmark
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

    # Resize frame
    img = cv2.resize(img, (new_width, new_height))

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 8)

    # Show the final output
    cv2.imshow("Image", img)
    if cv2.waitKey(7) & 0xFF == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
cam.release()
cv2.destroyAllWindows()
