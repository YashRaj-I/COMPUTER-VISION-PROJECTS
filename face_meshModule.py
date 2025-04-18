import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self,staticmode =False ,max_faces=5, min_detection_confidence=0.5,
                  min_tracking_confidence=0.5):
        self.staticmode = staticmode
        self.max_faces = max_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticmode, 
                                         max_num_faces=self.max_faces, 
                                         min_detection_confidence=self.min_detection_confidence, 
                                         min_tracking_confidence=self.min_tracking_confidence)

        self.Drawspec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
    
    def find_face_mesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                                self.Drawspec, self.Drawspec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id, cx, cy)
                    face.append([cx, cy])
                faces.append(face)
                #     # cv2.circle(img, (cx, cy), 1, (255, 0, 255), cv2.FILLED)
        return img, faces
        
        

    

def main():
        
    cam = cv2.VideoCapture(r"C:\Users\Asus\Documents\HAND_TRACKING\videos.mp4\6.mp4")
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cam.read()
        if not success:
            break
        img = cv2.resize(img, (840,780))
        img, faces = detector.find_face_mesh(img)
        if len(faces) != 0:
            print(len(faces))
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 8)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()