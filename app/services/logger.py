import logging
import os

if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename="logs/events.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

def log_event(message):
    logging.info(message)