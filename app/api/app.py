from flask import Flask, Response, jsonify, render_template, request
import cv2
import numpy as np
import threading
import time

app = Flask(__name__, template_folder="../frontend")

latest_frame = None
lock = threading.Lock()

current_status = {
    "status": "AWAKE",
    "attention": "FOCUSED",
    "fatigue_score": 0
}

score_history = []
event_log = []

# -------------------------------
# Receive frame from main.py
# -------------------------------
@app.route('/update_frame', methods=['POST'])
def update_frame():
    global latest_frame

    try:
        nparr = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        with lock:
            latest_frame = frame

        print("Frame received")  # DEBUG

        return "OK"
    except Exception as e:
        print("Error:", e)
        return "Error"


# -------------------------------
# Video streaming
# -------------------------------
def generate_frames():
    global latest_frame

    while True:
        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n')

        time.sleep(0.03)


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# -------------------------------
# Dashboard
# -------------------------------
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/status')
def status():
    return jsonify(current_status)


@app.route('/history')
def history():
    return jsonify(score_history)


@app.route('/events')
def events():
    return jsonify(event_log)


# -------------------------------
# Update status from main.py
# -------------------------------
def update_status(status, attention, score):
    current_status["status"] = status
    current_status["attention"] = attention
    current_status["fatigue_score"] = score

    score_history.append(score)
    if len(score_history) > 30:
        score_history.pop(0)

    timestamp = time.strftime("%H:%M:%S")
    event_log.append(f"[{timestamp}] {status}")

    if len(event_log) > 30:
        event_log.pop(0)


if __name__ == "__main__":
    app.run(debug=False, threaded=True)