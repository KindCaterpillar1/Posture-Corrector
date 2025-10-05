import cv2
import mediapipe as mp
import math
from playsound2 import playsound
import time

# ---- Mediapipe setup ----
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ---- Function to calculate angle between three points ----
def calculate_angle(a, b, c):
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]

    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians * 180.0 / math.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# ---- Video capture ----
cap = cv2.VideoCapture(0)

# ---- Timer for sound alert ----
last_alert = 0
ALERT_DELAY = 5  # seconds between alerts

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates for shoulders and hips
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

            # Calculate back angle (average of left and right)
            left_angle = calculate_angle(left_shoulder, left_hip, [left_hip.x, left_hip.y - 0.1])
            right_angle = calculate_angle(right_shoulder, right_hip, [right_hip.x, right_hip.y - 0.1])
            angle = (left_angle + right_angle) / 2

            # Display angle
            cv2.putText(image, str(int(angle)), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Check posture and play alert if slouching
            if angle < 155 and time.time() - last_alert > ALERT_DELAY:
                cv2.putText(image, 'Sit up straight!', (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                playsound("alert.mp3")
                last_alert = time.time()

        except:
            pass

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Posture Corrector', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

