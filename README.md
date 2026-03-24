# 🚗 AI Driver Monitoring System

This project is a real-time driver monitoring system built using computer vision.
It detects driver fatigue, drowsiness, yawning, distraction, and face occlusion, and provides live feedback through a dashboard along with alerts.

The system is designed to simulate a real-world driver safety solution and has been containerized using Docker for easy deployment.



## 🚀 What This Project Does

* Monitors driver behavior in real-time using webcam input
* Detects eye closure (drowsiness) and yawning
* Tracks head movement to identify distraction
* Handles face occlusion (e.g., hand covering face or no face visible)
* Calculates a fatigue score based on multiple factors with smoothing
* Sends live updates to a web dashboard
* Triggers alerts when unsafe conditions are detected
* Captures screenshots and records video during critical situations
* Plays alert sound via browser (Docker-compatible solution)



## 🧠 Core Logic (Simple Explanation)

* If eyes remain closed for a certain duration → system marks as **drowsy**
* If mouth is open frequently → system detects **yawning**
* If head is not facing forward → system detects **distraction**
* If face is not visible → system triggers **critical alert**
* Based on all signals → a **fatigue score (0–100)** is calculated
* If score crosses threshold → alert is triggered



## 🖥️ Dashboard

The system includes a web-based dashboard that shows:

* Live video feed
* Current driver status (Normal / Warning / Critical)
* Fatigue score
* Event logs
* Fatigue graph visualization
* Visual and audio alerts



## 🛠️ Tech Used

* Python
* OpenCV
* MediaPipe
* NumPy, SciPy
* Flask (for dashboard & APIs)
* JavaScript (Chart.js)
* Docker & Docker Compose



## 📦 Project Structure

```
driver-monitoring-system/
│
├── app/
│   ├── api/            # Flask backend
│   ├── detection/      # Drowsiness & head pose logic
│   ├── services/       # Alerts, recording, behavior tracking
│   ├── frontend/       # Dashboard UI
│   └── static/         # Static files (alarm audio)
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```



## ⚙️ How to Run (Docker)

git clone https://github.com/omugale21/ai-driver-monitoring-system.git  
cd ai-driver-monitoring-system  
docker compose up --build  

Then open:  
http://localhost:5000  



## ⚠️ Notes

* The system uses your local webcam
* Works best on Linux environments
* When running in Docker, video is shown via dashboard (not popup window)
* Audio alerts are handled in the browser due to Docker limitations
* Click once on the dashboard to enable sound alerts



## 🎥 Demo

I will be adding a demo video and screenshots soon.



## 🔮 Future Improvements

* Cloud-compatible version (upload-based demo)
* Better fatigue scoring model
* YOLO-based distraction detection
* Mobile/edge deployment
* API-based integration



## 📌 Final Thoughts

This project helped me understand:

* Real-time computer vision pipelines
* System design for AI applications
* Docker-based deployment
* Handling real-world challenges (occlusion, browser audio)
* Building a complete end-to-end application (not just a model)
