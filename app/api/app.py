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
# Receive frame
# -------------------------------
@app.route('/update_frame', methods=['POST'])
def update_frame():
    global latest_frame

    try:
        nparr = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        with lock:
            latest_frame = frame

        return "OK"
    except Exception as e:
        print("Error:", e)
        return "Error"


# -------------------------------
#  NEW: Receive status via API
# -------------------------------
@app.route('/update_status', methods=['POST'])
def update_status_api():
    global current_status, score_history, event_log

    data = request.json

    status = data.get("status")
    attention = data.get("attention")
    score = data.get("fatigue_score")

    current_status["status"] = status
    current_status["attention"] = attention
    current_status["fatigue_score"] = score

    # Store history
    score_history.append(score)
    if len(score_history) > 30:
        score_history.pop(0)

    # Event log
    timestamp = time.strftime("%H:%M:%S")
    event_log.append(f"[{timestamp}] {status} | Score: {score}")

    if len(event_log) > 30:
        event_log.pop(0)

    return "OK"


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
# Dashboard APIs
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)