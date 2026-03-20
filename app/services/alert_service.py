import os
import time
from threading import Thread
from app.services.voice_service import speak_async

alarm_active = False
last_alert_time = 0

# 🔥 CONFIG
COOLDOWN = 5  # seconds between alerts
DROWSY_THRESHOLD = 10


def sound_alarm():
    global alarm_active
    alarm_active = True

    os.system("mpg123 alarm.mp3 > /dev/null 2>&1")

    alarm_active = False


def trigger_alert(counter, attention):

    global last_alert_time

    current_time = time.time()

    # 🔥 COOLDOWN CHECK
    if current_time - last_alert_time < COOLDOWN:
        return

    # -------------------------------
    # 🧠 ATTENTION ALERT
    # -------------------------------
    if attention != "FOCUSED":
        speak_async("Please focus on the road")
        last_alert_time = current_time
        return

    # -------------------------------
    # ⚠️ DROWSINESS LEVEL 1
    # -------------------------------
    if counter > 5 and counter <= DROWSY_THRESHOLD:
        speak_async("You seem tired, please stay alert")
        last_alert_time = current_time
        return

    # -------------------------------
    # 🚨 DROWSINESS LEVEL 2 (STRONG)
    # -------------------------------
    if counter > DROWSY_THRESHOLD:
        speak_async("Wake up! You are falling asleep")

        if not alarm_active:
            Thread(target=sound_alarm, daemon=True).start()

        last_alert_time = current_time