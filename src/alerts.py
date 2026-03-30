"""Alert system for audio beeps and SMS notifications."""

import cv2
import threading
import winsound
from datetime import datetime
from .config import Config


class AlertManager:
    """Manages audio and SMS alerts with rate limiting."""

    def __init__(self):
        """Initialize alert manager."""
        self.last_alert_time = {}
        self.alert_cooldown = Config.ALERT_COOLDOWN
        self.sms_enabled = Config.is_twilio_configured()

        # Alert types
        self.ALERT_DROWSY = "drowsy"
        self.ALERT_PHONE = "phone"
        self.ALERT_YAWN = "yawn"

    def trigger_audio_alert(self, alert_type, duration=500, frequency=1000):
        """
        Play an audio beep alert.

        Args:
            alert_type: Type of alert (drowsy, phone, yawn)
            duration: Beep duration in milliseconds
            frequency: Beep frequency in Hz
        """
        try:
            # Different patterns for different alerts
            if alert_type == self.ALERT_DROWSY:
                # Urgent double beep for drowsiness
                winsound.Beep(frequency, duration)
                winsound.Beep(frequency + 200, duration)
            elif alert_type == self.ALERT_PHONE:
                # Single beep for phone detection
                winsound.Beep(frequency - 100, duration)
            elif alert_type == self.ALERT_YAWN:
                # Softer beep for yawning
                winsound.Beep(frequency - 200, duration // 2)

        except Exception as e:
            print(f"Audio alert failed: {e}")

    def trigger_sms_alert(self, alert_type, driver_name="Driver"):
        """
        Send SMS alert via Twilio (non-blocking).

        Args:
            alert_type: Type of alert
            driver_name: Name of the driver for personalization
        """
        if not self.sms_enabled:
            return False

        # Check cooldown
        if not self._check_cooldown(alert_type):
            return False

        message = self._format_sms_message(alert_type, driver_name)

        # Send SMS in background thread
        thread = threading.Thread(
            target=self._send_sms,
            args=(message,)
        )
        thread.daemon = True
        thread.start()

        return True

    def _check_cooldown(self, alert_type):
        """Check if alert is within cooldown period."""
        now = datetime.now()

        if alert_type in self.last_alert_time:
            elapsed = (now - self.last_alert_time[alert_type]).total_seconds()
            # Cooldown is in frames, convert to seconds (~30fps)
            cooldown_seconds = self.alert_cooldown / 30.0
            if elapsed < cooldown_seconds:
                return False

        self.last_alert_time[alert_type] = now
        return True

    def _format_sms_message(self, alert_type, driver_name):
        """Format SMS message based on alert type."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        messages = {
            self.ALERT_DROWSY: (
                f"⚠️ DROWSINESS ALERT\n"
                f"Driver: {driver_name}\n"
                f"Time: {timestamp}\n"
                f"Please take a break!"
            ),
            self.ALERT_PHONE: (
                f"📱 PHONE USAGE ALERT\n"
                f"Driver: {driver_name}\n"
                f"Time: {timestamp}\n"
                f"Focus on driving!"
            ),
            self.ALERT_YAWN: (
                f"😴 FATIGUE DETECTED\n"
                f"Driver: {driver_name}\n"
                f"Time: {timestamp}\n"
                f"Consider resting soon."
            )
        }

        return messages.get(alert_type, f"Alert: {alert_type} at {timestamp}")

    def _send_sms(self, message):
        """Send SMS via Twilio."""
        try:
            from twilio.rest import Client

            client = Client(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)

            client.messages.create(
                body=message,
                from_=Config.TWILIO_PHONE_NUMBER,
                to=Config.ALERT_RECIPIENT_NUMBER
            )
            print(f"SMS sent successfully")

        except Exception as e:
            print(f"Failed to send SMS: {e}")

    def reset_cooldown(self, alert_type=None):
        """Reset cooldown for specific or all alerts."""
        if alert_type:
            if alert_type in self.last_alert_time:
                del self.last_alert_time[alert_type]
        else:
            self.last_alert_time = {}


class VisualAlert:
    """Visual alert overlays for the video frame."""

    @staticmethod
    def draw_alert_banner(frame, alert_type, level=2):
        """
        Draw visual alert banner on frame.

        Args:
            frame: Video frame
            alert_type: Type of alert
            level: Alert level (1=warning, 2=critical)
        """
        overlay = frame.copy()

        # Colors based on level
        if level == 2:
            color = (0, 0, 255)  # Red for critical
        else:
            color = (0, 165, 255)  # Orange for warning

        # Banner text
        banners = {
            "drowsy": "⚠️ DROWSY - PULL OVER!",
            "phone": "📱 PUT PHONE AWAY!",
            "yawn": "😴 TIRED?",
            "normal": ""
        }

        text = banners.get(alert_type, "")
        if not text:
            return frame

        # Draw banner at top
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw text
        cv2.putText(frame, text, (frame.shape[1] // 2 - 150, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        return frame

    @staticmethod
    def draw_status_panel(frame, status_dict):
        """
        Draw status panel with all detection states.

        Args:
            frame: Video frame
            status_dict: Dictionary of status values
        """
        y_offset = 30
        x_offset = frame.shape[1] - 250

        for key, value in status_dict.items():
            color = (0, 255, 0) if value else (0, 0, 255)
            status = "✓" if value else "✗"

            cv2.putText(frame, f"{key}: {status}", (x_offset, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25

        return frame
