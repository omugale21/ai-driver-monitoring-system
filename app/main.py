import cv2
import mediapipe as mp
import numpy as np
import requests
import time
import os

from app.detection.drowsiness import DrowsinessDetector
from app.detection.head_pose import HeadPoseEstimator, get_attention
from app.services.alert_service import trigger_alert
from app.services.fatigue_service import FatigueScorer
from app.services.camera_service import WebcamStream
from app.services.recording_service import VideoRecorder
from app.services.behavior_service import BehaviorMonitor
from app.services.voice_service import speak_async
from app.services.screenshot_service import ScreenshotService
from app.config import *

# -------------------------------
# Initialize
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False
)

drowsy_detector = DrowsinessDetector()
head_pose = HeadPoseEstimator()
fatigue_scorer = FatigueScorer()
recorder = VideoRecorder()
behavior = BehaviorMonitor()
screenshot = ScreenshotService()

cap = WebcamStream(src=0).start()

# Session tracking
max_fatigue = 0
total_alerts = 0
start_time = time.time()

frame_count = 0
process_counter = 0

print("Press 'q' to exit")

# -------------------------------
# Main Loop
# -------------------------------
while True:

    frame = cap.read()
    if frame is None:
        continue

    frame = cv2.resize(frame, (640, 480))
    frame = cv2.flip(frame, 1)

    process_counter += 1

    # Optimization: skip alternate frames
    if process_counter % 2 != 0:
        if os.environ.get("DISPLAY"):
            cv2.imshow("Driver Monitoring", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        continue

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

            # Detection
            status, ear, mar, _ = drowsy_detector.process(points)

            pitch, yaw, roll = head_pose.estimate(frame, points)
            attention = get_attention(pitch, yaw, head_pose, points, frame.shape)

            fatigue_score = fatigue_scorer.update(status, attention, mar)

            # Session tracking
            max_fatigue = max(max_fatigue, fatigue_score)

            if fatigue_score > FATIGUE_WARNING:
                total_alerts += 1

            # Alerts
            trigger_alert(fatigue_score, attention)

            # Behavior
            is_distracted = behavior.check_distraction(attention)
            is_microsleep = behavior.check_microsleep(status)

            if is_distracted:
                speak_async("Driver distracted")

            if is_microsleep:
                speak_async("Micro sleep detected")

            # Recording
            if RECORDING_ENABLED and fatigue_score > FATIGUE_CRITICAL:
                recorder.start(frame)
                recorder.write(frame)
            else:
                recorder.stop()

            # Screenshot
            if SCREENSHOT_ENABLED and fatigue_score > FATIGUE_SCREENSHOT:
                screenshot.capture(frame)

            # -------------------------------
            # UI Overlay
            # -------------------------------

            if fatigue_score < 40:
                color = (0, 255, 0)
            elif fatigue_score < 60:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)

            display_attention = "FOCUSED" if attention == "FOCUSED" else "DISTRACTED"

            cv2.putText(frame, f"Status: {status}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.putText(frame, f"Attention: {display_attention}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.putText(frame, f"Score: {fatigue_score}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # -------------------------------
    #  DOCKER FIX (IMPORTANT)
    # -------------------------------
    frame_count += 1

    if frame_count % 5 == 0:
        _, jpeg = cv2.imencode('.jpg', frame)

        try:
            requests.post(
                "http://flask:5000/update_frame",   # ✅ FIXED
                data=jpeg.tobytes(),
                timeout=0.1
            )
        except:
            pass

        try:
            requests.post(
                "http://flask:5000/update_status",  # ✅ FIXED
                json={
                    "status": status,
                    "attention": attention,
                    "fatigue_score": fatigue_score
                },
                timeout=0.1
            )
        except:
            pass

    # Show window only in local system
    if os.environ.get("DISPLAY"):
        cv2.imshow("Driver Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# -------------------------------
# Session Summary
# -------------------------------
total_time = int(time.time() - start_time)

print("\n===== SESSION SUMMARY =====")
print(f"Total Time: {total_time} sec")
print(f"Max Fatigue Score: {max_fatigue}")
print(f"Total Alerts: {total_alerts}")
print("===========================\n")

cap.stop()
cv2.destroyAllWindows()