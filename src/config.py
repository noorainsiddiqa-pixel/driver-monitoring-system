"""Configuration settings for the driver monitoring system."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Centralized configuration management."""

    # Detection thresholds
    EAR_THRESHOLD = float(os.getenv('EAR_THRESHOLD', 0.28))
    PHONE_CONFIDENCE_THRESHOLD = float(os.getenv('PHONE_CONFIDENCE_THRESHOLD', 0.5))
    DROWSINESS_FRAMES = int(os.getenv('DROWSINESS_FRAMES', 15))  # ~0.5 seconds at 30fps
    NIGHT_VISION_THRESHOLD = int(os.getenv('NIGHT_VISION_THRESHOLD', 50))

    # Timing (in frames at ~30fps)
    ALERT_COOLDOWN = 30  # Frames between repeated alerts
    PERSISTENCE_FRAMES = 5  # Frames to confirm detection

    # Twilio configuration
    TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
    TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
    TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
    ALERT_RECIPIENT_NUMBER = os.getenv('ALERT_RECIPIENT_NUMBER')

    # Display settings
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2

    # Colors (BGR)
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_BLUE = (255, 0, 0)
    COLOR_WHITE = (255, 255, 255)

    @classmethod
    def is_twilio_configured(cls):
        """Check if Twilio is properly configured."""
        return all([
            cls.TWILIO_ACCOUNT_SID,
            cls.TWILIO_AUTH_TOKEN,
            cls.TWILIO_PHONE_NUMBER,
            cls.ALERT_RECIPIENT_NUMBER
        ])


class EyeLandmarks:
    """Eye landmark indices for MediaPipe Face Mesh."""
    # Left eye
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    LEFT_EYE_LEFT = 33
    LEFT_EYE_RIGHT = 133
    LEFT_EYE_UPPER_LID = 157
    LEFT_EYE_LOWER_LID = 163

    # Right eye
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    RIGHT_EYE_LEFT = 362
    RIGHT_EYE_RIGHT = 263
    RIGHT_EYE_UPPER_LID = 385
    RIGHT_EYE_LOWER_LID = 398

    # Mouth landmarks for yawning detection
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
