import cv2
import mediapipe as mp
import math
import pygame

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- Function to play alert sound ---
def play_alert():
    pygame.mixer.init()
    if not pygame.mixer.music.get_busy():  # prevent overlap
        pygame.mixer.music.load("alert.wav")  # or .wav if that’s your file
        pygame.mixer.music.play()

# --- Function to calculate body angle ---
def calculate_angle(a, b, c):
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) -
        math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return abs(ang)

cap = cv2.VideoCapture(0)

with mp_pose.Pose() as pose:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            angle = calculate_angle(left_ear, left_shoulder, left_hip)

            cv2.putText(frame, f'Angle: {int(angle)}', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#edit angle it warns however you want
            if angle < 173:
                cv2.putText(frame, 'Sit up straight!', (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                play_alert()  # play the sound alert

        cv2.imshow('Posture AI', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
