import cv2
import mediapipe as mp
import time

# cam = cv2.VideoCapture(r"C:\Users\Asus\Documents\HAND_TRACKING\videos.mp4\2.mp4")
cam = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=5)
Drawspec = mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
while True:
    success, img = cam.read()
    if not success:
        break
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, Drawspec, Drawspec)
            for id, lm in enumerate(faceLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
            #     cv2.circle(img, (cx, cy), 1, (255, 0, 255), cv2.FILLED)
    img = cv2.resize(img, (840,780))
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 8)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break