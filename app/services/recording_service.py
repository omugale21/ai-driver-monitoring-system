import cv2
import os
import time


class VideoRecorder:
    def __init__(self):
        self.recording = False
        self.out = None

        self.prev_time = None
        self.fps = 10  # default fallback

        if not os.path.exists("recordings"):
            os.makedirs("recordings")

    def start(self, frame):
        if self.recording:
            return

        timestamp = time.strftime("%H_%M_%S")
        filename = f"recordings/drowsy_{timestamp}.avi"

        h, w, _ = frame.shape

        #  Use MJPG codec
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        self.out = cv2.VideoWriter(filename, fourcc, self.fps, (w, h))

        self.recording = True
        self.prev_time = time.time()

        print(f"[INFO] Recording started: {filename}")

    def write(self, frame):
        if not self.recording or self.out is None:
            return

        #  Measure actual FPS dynamically
        current_time = time.time()
        if self.prev_time is not None:
            time_diff = current_time - self.prev_time

            if time_diff > 0:
                self.fps = 1.0 / time_diff

        self.prev_time = current_time

        # Write frame
        self.out.write(frame)

    def stop(self):
        if self.recording and self.out is not None:
            self.out.release()
            self.out = None
            self.recording = False
            self.prev_time = None

            print("[INFO] Recording stopped")