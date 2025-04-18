import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define custom colors (B, G, R format)
LINE_COLOR = (0, 255, 0)  # Green
POINT_COLOR = (0, 0, 255)  # Red

# Custom drawing styles
hand_landmark_style = mp_drawing.DrawingSpec(color=POINT_COLOR, thickness=5)
hand_connection_style = mp_drawing.DrawingSpec(color=LINE_COLOR, thickness=3)

cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    hand_landmark_style, hand_connection_style  # Apply custom styles
                )

        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
