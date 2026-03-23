# -------------------------------
#  EXISTING (KEEP THESE)
# -------------------------------
EAR_THRESHOLD = 0.23
MAR_THRESHOLD = 0.75
CONSEC_FRAMES = 15

LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))
MOUTH = list(range(48, 68))


# -------------------------------
#  NEW (ADVANCED CONFIG)
# -------------------------------

# Fatigue thresholds
FATIGUE_WARNING = 40
FATIGUE_CRITICAL = 60
FATIGUE_SCREENSHOT = 70

# Time thresholds (seconds)
DISTRACTION_TIME = 3
MICROSLEEP_TIME = 2

# Feature toggles
RECORDING_ENABLED = True
SCREENSHOT_ENABLED = True