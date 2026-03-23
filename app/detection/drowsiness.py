import numpy as np
from scipy.spatial import distance

class DrowsinessDetector:

    def __init__(self):
        self.counter = 0  # counts consecutive drowsy frames

    # -------------------------------
    # Eye Aspect Ratio (EAR)
    # -------------------------------
    def calculate_EAR(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    # -------------------------------
    # Mouth Aspect Ratio (MAR)
    # -------------------------------
    def calculate_MAR(self, mouth):
        A = distance.euclidean(mouth[1], mouth[7])
        B = distance.euclidean(mouth[3], mouth[5])
        C = distance.euclidean(mouth[0], mouth[4])
        return (A + B) / (2.0 * C)

    # -------------------------------
    # Main Detection Logic
    # -------------------------------
    def process(self, points):

        #  MediaPipe Landmark Indices (stable selection)
        LEFT_EYE = [362, 385, 387, 263, 373, 380]
        RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]

        # Extract regions
        left_eye = points[LEFT_EYE]
        right_eye = points[RIGHT_EYE]
        mouth = points[MOUTH]

        # Calculate EAR & MAR
        ear = (self.calculate_EAR(left_eye) +
               self.calculate_EAR(right_eye)) / 2.0

        mar = self.calculate_MAR(mouth)

        # -------------------------------
        #  TUNED THRESHOLDS (MediaPipe)
        # -------------------------------
        EAR_THRESHOLD = 0.25
        MAR_THRESHOLD = 0.65

        fatigue_score = 0

        # -------------------------------
        # Eye-based detection
        # -------------------------------
        if ear < EAR_THRESHOLD:
            self.counter += 1
            fatigue_score += 60
        else:
            self.counter = 0

        # -------------------------------
        # Mouth-based detection
        # -------------------------------
        if mar > MAR_THRESHOLD:
            fatigue_score += 40

        # Clamp score
        fatigue_score = min(100, fatigue_score)

        # -------------------------------
        # Final Status
        # -------------------------------
        if self.counter > 10:
            status = "DROWSY"
        else:
            status = "AWAKE"

        return status, ear, mar, fatigue_score