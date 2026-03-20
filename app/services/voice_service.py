import pyttsx3
from threading import Thread

# Initialize engine
engine = pyttsx3.init()

# 🔥 Configure voice
engine.setProperty('rate', 150)  # Speed
engine.setProperty('volume', 1.0)  # Max volume


def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except:
        pass


def speak_async(text):
    Thread(target=speak, args=(text,), daemon=True).start()