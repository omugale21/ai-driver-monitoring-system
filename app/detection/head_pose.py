import cv2
import numpy as np
from collections import deque

class HeadPoseEstimator:
    def __init__(self):
        self.model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ])

        # smoothing (still useful for yaw)
        self.pitch_history = deque(maxlen=5)
        self.yaw_history = deque(maxlen=5)

    def estimate(self, frame, landmarks):
        size = frame.shape

        image_points = np.array([
            landmarks[1],
            landmarks[152],
            landmarks[33],
            landmarks[263],
            landmarks[61],
            landmarks[291]
        ], dtype="double")

        focal_length = size[1]
        center = (size[1] // 2, size[0] // 2)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))

        _, rotation_vector, _ = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs
        )

        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch, yaw, roll = angles

        # smoothing
        self.pitch_history.append(pitch)
        self.yaw_history.append(yaw)

        pitch = np.mean(self.pitch_history)
        yaw = np.mean(self.yaw_history)

        return pitch, yaw, roll


# 🔥 FINAL ATTENTION FUNCTION (NOSE-BASED — BEST)
def get_attention(pitch, yaw, estimator, points, frame_shape):

    h, w, _ = frame_shape

    # Nose landmark (MediaPipe index 1)
    nose_x, nose_y = points[1]

    # Frame center
    center_x = w // 2
    center_y = h // 2

    # Thresholds (adjustable if needed)
    X_THRESHOLD = w * 0.15
    Y_THRESHOLD = h * 0.15

    # LEFT / RIGHT
    if nose_x < center_x - X_THRESHOLD:
        return "LOOKING_LEFT"

    elif nose_x > center_x + X_THRESHOLD:
        return "LOOKING_RIGHT"

    # UP / DOWN
    if nose_y < center_y - Y_THRESHOLD:
        return "LOOKING_UP"

    elif nose_y > center_y + Y_THRESHOLD:
        return "LOOKING_DOWN"

    # 🔥 PERFECT FOCUSED
    return "FOCUSED"