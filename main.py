import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from threading import Thread
import os
from collections import deque
import time

# -------------------------------
# Webcam Stream Class
# -------------------------------
class WebcamStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()


# -------------------------------
# EAR (Eye)
# -------------------------------
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


# -------------------------------
# MAR (Mouth)
# -------------------------------
def calculate_MAR(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)


# -------------------------------
# Alarm
# -------------------------------
def sound_alarm():
    os.system("mpg123 alarm.mp3")


# -------------------------------
# Load Models
# -------------------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# -------------------------------
# Constants
# -------------------------------
EAR_THRESHOLD = 0.23
MAR_THRESHOLD = 0.75

CONSEC_FRAMES = 15

counter = 0
yawn_counter = 0

status = "Awake"

LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))
MOUTH = list(range(48, 68))

ear_history = deque(maxlen=10)

prev_time = time.time()

# -------------------------------
# Start Webcam
# -------------------------------
stream = WebcamStream().start()

print("Press 'q' to exit")

# -------------------------------
# Main Loop
# -------------------------------
while True:
    frame = stream.read()
    if frame is None:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        landmarks = predictor(gray, face)
        points = np.array([[p.x, p.y] for p in landmarks.parts()])

        left_eye = points[LEFT_EYE]
        right_eye = points[RIGHT_EYE]
        mouth = points[MOUTH]

        left_EAR = calculate_EAR(left_eye)
        right_EAR = calculate_EAR(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        ear_history.append(avg_EAR)
        smooth_EAR = np.mean(ear_history)

        MAR = calculate_MAR(mouth)

        # Draw
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [mouth], True, (0, 255, 255), 1)

        # -------------------------------
        # Detection Logic
        # -------------------------------
        if smooth_EAR < EAR_THRESHOLD:
            counter += 1
        else:
            counter = 0

        if MAR > MAR_THRESHOLD:
            yawn_counter += 1
        else:
            yawn_counter = 0

        # Decision
        if counter >= CONSEC_FRAMES and yawn_counter >= CONSEC_FRAMES:
            status = "SLEEP + YAWN"
            color = (0, 0, 255)

        elif counter >= CONSEC_FRAMES:
            status = "DROWSY"
            color = (0, 0, 255)

        elif yawn_counter >= CONSEC_FRAMES:
            status = "YAWNING"
            color = (0, 255, 255)

        else:
            status = "AWAKE"
            color = (0, 255, 0)

        # Alert
        if status != "AWAKE":
            cv2.putText(frame, "ALERT!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            if counter % 20 == 0:
                Thread(target=sound_alarm, daemon=True).start()

        # Display Info
        cv2.putText(frame, f"EAR: {smooth_EAR:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"MAR: {MAR:.2f}", (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Status: {status}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (500, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Driver Fatigue Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

stream.stop()
cv2.destroyAllWindows()