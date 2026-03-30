"""Phone detection using YOLOv8."""

import cv2

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    YOLO = None
    print("⚠️ Warning: ultralytics package not installed. Phone detection disabled.")

from .config import Config


class PhoneDetector:
    """Real-time phone detection using YOLOv8."""

    # COCO dataset classes - phone is class 67
    PHONE_CLASS_ID = 67
    PHONE_CLASS_NAME = 'cell phone'

    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize YOLOv8 phone detector.

        Args:
            model_path: Path to YOLOv8 model weights
        """
        if YOLO is None:
            print("⚠️ Phone detection disabled because YOLO is unavailable.")
            self.model = None
            self.model_loaded = False
        else:
            try:
                self.model = YOLO(model_path)
                self.model_loaded = True
            except Exception as e:
                print(f"Warning: Could not load YOLO model: {e}")
                self.model = None
                self.model_loaded = False

        # Detection history for temporal consistency
        self.detection_history = []
        self.history_size = Config.PERSISTENCE_FRAMES

    def detect(self, frame):
        """
        Detect phone in frame.

        Args:
            frame: Input video frame

        Returns:
            tuple: (is_phone_detected, confidence, frame, phone_location)
        """
        if not self.model_loaded:
            return False, 0.0, frame, None

        # Run inference
        results = self.model(frame, verbose=False, conf=Config.PHONE_CONFIDENCE_THRESHOLD)
        result = results[0]

        is_detected = False
        max_confidence = 0.0
        phone_location = None

        # Process detections
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # Check if it's a cell phone
                if cls_id == self.PHONE_CLASS_ID:
                    if conf > max_confidence:
                        max_confidence = conf
                        is_detected = True

                        # Get bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        phone_location = (x1, y1, x2, y2)

        # Add to history for temporal smoothing
        self.detection_history.append(is_detected)
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)

        # Require consistent detection
        stable_detection = sum(self.detection_history) >= (self.history_size // 2 + 1)
        is_detected = is_detected and stable_detection

        # Draw detection
        frame = self._draw_detection(frame, is_detected, max_confidence, phone_location)

        return is_detected, max_confidence, frame, phone_location

    def _draw_detection(self, frame, is_detected, confidence, bbox):
        """Draw phone detection results on frame."""
        if bbox is None:
            return frame

        x1, y1, x2, y2 = bbox

        # Choose color based on detection status
        color = (0, 0, 255) if is_detected else (0, 255, 0)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"Phone: {confidence:.2f}" if is_detected else "No phone"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Add warning overlay if phone detected
        if is_detected:
            # Semi-transparent red overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1 - 5, y1 - 25), (x2 + 5, y2 + 5), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        return frame

    def release(self):
        """Release resources."""
        self.model = None
        self.model_loaded = False
