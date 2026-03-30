"""
Driver Monitoring System
========================
Real-time driver drowsiness and phone usage detection with SMS alerts.

Features:
- EAR-based drowsiness detection
- YOLOv8 phone detection
- Yawning detection
- Night vision enhancement
- Audio and SMS alerts
"""

import sys
import time
import logging
import cv2
import mediapipe as mp
from collections import deque

# Add src to path
sys.path.insert(0, '.')

from src.config import Config
from src.drowsiness import detect_drowsiness, detect_yawn
from src.phone_detect import PhoneDetector
from src.alerts import AlertManager, VisualAlert
from src.utils import (
    enhance_low_light,
    apply_night_vision_indicator,
    resize_frame,
    draw_fps_counter,
    draw_detector_status
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('driver_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DriverMonitor:
    """Main driver monitoring system."""

    def __init__(self):
        """Initialize the driver monitoring system."""
        logger.info("Initializing Driver Monitoring System...")

        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Phone detector
        self.phone_detector = PhoneDetector()

        # Alert manager
        self.alert_manager = AlertManager()

        # State tracking
        self.ear_history = deque(maxlen=Config.DROWSINESS_FRAMES)
        self.frame_count = 0
        self.start_time = time.time()

        # Detection state
        self.is_drowsy = False
        self.is_yawning = False
        self.phone_detected = False
        self.face_detected = False

        logger.info("Initialization complete")

    def process_frame(self, frame):
        """
        Process a single frame for all detections.

        Args:
            frame: Input video frame

        Returns:
            Processed frame with annotations
        """
        self.frame_count += 1

        # Enhance low-light conditions
        frame, is_night_mode = enhance_low_light(frame)
        frame = apply_night_vision_indicator(frame, is_night_mode)

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        # Check face detection
        self.face_detected = results.multi_face_landmarks is not None

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # Drowsiness detection
            self.is_drowsy, drowsiness_level, self.ear_history, frame = detect_drowsiness(
                frame, face_landmarks, self.ear_history, self.frame_count
            )

            # Yawning detection
            self.is_yawning, mouth_ratio, frame = detect_yawn(frame, face_landmarks)

            # Trigger alerts for drowsiness
            if self.is_drowsy and drowsiness_level >= 2:
                self.alert_manager.trigger_audio_alert(
                    self.alert_manager.ALERT_DROWSY
                )
                self.alert_manager.trigger_sms_alert(
                    self.alert_manager.ALERT_DROWSY
                )
                frame = VisualAlert.draw_alert_banner(frame, "drowsy", level=2)

            # Trigger alert for yawning
            if self.is_yawning:
                self.alert_manager.trigger_audio_alert(
                    self.alert_manager.ALERT_YAWN
                )
                frame = VisualAlert.draw_alert_banner(frame, "yawn", level=1)
        else:
            # No face detected - reset state
            self.is_drowsy = False
            self.is_yawning = False

        # Phone detection (runs independently of face detection)
        self.phone_detected, phone_conf, frame, _ = self.phone_detector.detect(frame)

        if self.phone_detected:
            self.alert_manager.trigger_audio_alert(
                self.alert_manager.ALERT_PHONE
            )
            self.alert_manager.trigger_sms_alert(
                self.alert_manager.ALERT_PHONE
            )
            frame = VisualAlert.draw_alert_banner(frame, "phone", level=2)

        # Draw status panel
        status = {
            "Drowsy": self.is_drowsy,
            "Yawning": self.is_yawning,
            "Phone": self.phone_detected
        }
        frame = VisualAlert.draw_status_panel(frame, status)

        # Draw FPS and detector status
        fps = self.frame_count / max(time.time() - self.start_time, 0.001)
        frame = draw_fps_counter(frame, fps)
        frame = draw_detector_status(frame, self.face_detected, self.phone_detector.model_loaded)

        return frame

    def run(self, camera_source=0):
        """
        Run the main monitoring loop.

        Args:
            camera_source: Camera device index or video file path
        """
        logger.info(f"Opening camera: {camera_source}")

        cap = cv2.VideoCapture(camera_source)

        if not cap.isOpened():
            logger.error("Failed to open camera")
            print("❌ Error: Could not open camera")
            print("   Check if camera is connected or try a different camera index.")
            return

        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.WINDOW_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.WINDOW_HEIGHT)

        logger.info("Camera opened successfully")
        print("✅ Driver Monitoring System started")
        print("   Press 'q' to quit")
        print("   Press 's' to send test SMS")
        print("   Press 'r' to reset alert cooldowns")

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    logger.warning("Failed to read frame")
                    print("⚠️ Frame not received - check camera connection")
                    break

                # Process frame
                frame = self.process_frame(frame)

                # Display
                cv2.imshow("Driver Monitoring System", frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    logger.info("User requested exit")
                    break
                elif key == ord('s'):
                    # Send test SMS
                    print("📨 Sending test SMS...")
                    self.alert_manager.trigger_sms_alert(
                        self.alert_manager.ALERT_DROWSY,
                        driver_name="Test"
                    )
                elif key == ord('r'):
                    # Reset cooldowns
                    self.alert_manager.reset_cooldown()
                    print("🔄 Alert cooldowns reset")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.phone_detector.release()
            logger.info("System shutdown complete")

        print("🛑 Driver Monitoring System stopped")

    def cleanup(self):
        """Clean up resources."""
        self.face_mesh.close()
        self.phone_detector.release()
        cv2.destroyAllWindows()


def main():
    """Entry point."""
    print("=" * 50)
    print("  DRIVER MONITORING SYSTEM")
    print("=" * 50)

    # Check Twilio configuration
    if not Config.is_twilio_configured():
        print("⚠️  SMS alerts disabled - configure .env for Twilio")
        print("   Copy .env.example to .env and add your credentials")
    else:
        print("✅ SMS alerts enabled")

    print()

    monitor = DriverMonitor()

    try:
        monitor.run()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
    finally:
        monitor.cleanup()


if __name__ == "__main__":
    main()
