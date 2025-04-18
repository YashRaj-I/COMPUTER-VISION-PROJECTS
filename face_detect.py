import cv2
import time
import mediapipe as mp

# cam = cv2.VideoCapture(r"C:\Users\Asus\Documents\HAND_TRACKING\videos.mp4\4.mp4")
cam = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path
pTime = 0

mpFace = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFace.FaceDetection(0.75)
while True:
    success, img = cam.read()
    if not success:
        break  # Exit if video ends or can't read frame
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    if results.detections:
        for id, detection in enumerate(results.detections):
            mpDraw.draw_detection(img, detection)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(img, f'Face {int(detection.score[0]*100)}%', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 5,
                        (0, 255, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 3)


    # Resize the image to 1080x780
    img = cv2.resize(img, (1000,900))  

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Quit when 'q' is pressed