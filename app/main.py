import cv2
import mediapipe as mp
import numpy as np
import requests
import time

from app.detection.drowsiness import DrowsinessDetector
from app.detection.head_pose import HeadPoseEstimator, get_attention
from app.services.alert_service import trigger_alert
from app.services.camera_service import WebcamStream
from app.api.app import update_status

# -------------------------------
# Initialize
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

drowsy_detector = DrowsinessDetector()
head_pose = HeadPoseEstimator()

# 🔥 Threaded camera (better FPS)
cap = WebcamStream(src=0).start()

frame_count = 0

print("Press 'q' to exit")

# -------------------------------
# Main Loop
# -------------------------------
while True:

    frame = cap.read()
    if frame is None:
        continue

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status = "AWAKE"
    fatigue_score = 0
    attention = "FOCUSED"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            h, w, _ = frame.shape

            points = np.array([
                (int(lm.x * w), int(lm.y * h))
                for lm in face_landmarks.landmark
            ])

            # -------------------------------
            # 🔥 Drowsiness Detection
            # -------------------------------
            status, ear, mar, fatigue_score = drowsy_detector.process(points)

            # -------------------------------
            # 🔥 Head Pose / Attention Detection (NEW METHOD)
            # -------------------------------
            pitch, yaw, roll = head_pose.estimate(frame, points)
            attention = get_attention(pitch, yaw, head_pose, points, frame.shape)

            # -------------------------------
            # 🔥 SMART ALERT SYSTEM
            # -------------------------------
            trigger_alert(drowsy_detector.counter, attention)

            # -------------------------------
            # Draw UI
            # -------------------------------
            cv2.putText(frame, f"Status: {status}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if status == "AWAKE" else (0, 0, 255), 2)

            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f"MAR: {mar:.2f}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f"Fatigue: {fatigue_score}%", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.putText(frame, f"Attention: {attention}", (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 0), 2)

    # -------------------------------
    # Send frame to Flask (optimized)
    # -------------------------------
    frame_count += 1

    if frame_count % 3 == 0:
        _, jpeg = cv2.imencode('.jpg', frame)

        try:
            requests.post(
                "http://127.0.0.1:5000/update_frame",
                data=jpeg.tobytes(),
                timeout=0.1
            )
        except:
            pass

    # -------------------------------
    # Update status
    # -------------------------------
    update_status(status, attention, fatigue_score)

    # -------------------------------
    # Show window
    # -------------------------------
    cv2.imshow("Driver Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.01)

cap.stop()
cv2.destroyAllWindows()