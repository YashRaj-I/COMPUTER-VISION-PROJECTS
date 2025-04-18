import cv2
import time
import mediapipe as mp

class FaceDetector:
    def __init__(self, mindet_conf=0.75):
        self.mindet_conf = mindet_conf
        self.mpFace = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFace.FaceDetection(self.mindet_conf)

    def find_faces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxes = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                bboxes.append([id,bbox, detection.score])
                if draw:
                    img = self.FancyDraw(img, bbox)
                
                    cv2.putText(img, f'Face {int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                5, (0, 255, 0), 12)
        return img, bboxes
    def FancyDraw(self, img, bbox, l =30, t= 10, draw=True):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        cv2.rectangle(img, bbox, (255, 0, 255), 2)
        # Top right
        cv2.line(img, (x, y), (x+l, y), (255, 0, 255), t)  
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)  
        # Top left
        cv2.line(img, (x1, y), (x1-l, y), (255, 0, 255), t)  
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)
        # Bottom right
        cv2.line(img, (x, y1), (x+l, y1), (255, 0, 255), t)  
        cv2.line(img, (x, y1), (x, y1-l), (255, 0, 255), t)
        # Bottom left
        cv2.line(img, (x1, y1), (x1-l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1-l), (255, 0, 255), t)


        return img


def main():
    cam = cv2.VideoCapture(r"C:\Users\Asus\Documents\HAND_TRACKING\videos.mp4\4.mp4")
    pTime = time.time()  # Initialize pTime properly
    detector = FaceDetector()

    while True:
        success, img = cam.read()
        if not success:
            break  # Exit if video ends or can't read frame

        img, bboxes = detector.find_faces(img, draw=True)  # Set draw=False to avoid drawing in this step
        # print(bboxes) 
        # Correct FPS Calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime)  # Avoid division by zero
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        # Resize the image to an appropriate dimension
        img = cv2.resize(img, (880,680))  # Adjust the size as needed

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit when 'q' is pressed
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
