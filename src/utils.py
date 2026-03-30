"""Utility functions for image processing and helpers."""

import cv2
import numpy as np
from .config import Config


def enhance_low_light(frame, threshold=None):
    """
    Enhance low-light images using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        frame: Input video frame
        threshold: Brightness threshold below which enhancement is applied

    Returns:
        tuple: (enhanced_frame, is_night_mode)
    """
    if threshold is None:
        threshold = Config.NIGHT_VISION_THRESHOLD

    # Calculate average brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)

    is_night_mode = avg_brightness < threshold

    if not is_night_mode:
        return frame, False

    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Split the LAB channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel (lightness)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)

    # Merge channels back
    enhanced_lab = cv2.merge((enhanced_l, a, b))

    # Convert back to BGR
    enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Apply slight gamma correction for additional brightness
    gamma = 0.8
    inv_gamma = 1.0 / gamma
    enhanced_frame = np.array(
        [((v / 255.0) ** (1.0 / inv_gamma) * 255)
         for v in enhanced_frame.flatten().astype(float)]
    ).reshape(enhanced_frame.shape).astype(np.uint8)

    return enhanced_frame, is_night_mode


def apply_night_vision_indicator(frame, is_night_mode):
    """
    Add night vision indicator to frame.

    Args:
        frame: Video frame
        is_night_mode: Whether night mode is active

    Returns:
        Frame with indicator
    """
    if not is_night_mode:
        return frame

    # Draw night vision indicator
    cv2.putText(frame, "NIGHT VISION ON", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Add subtle green tint overlay
    overlay = frame.copy()
    cv2.addWeighted(overlay, 0.95,
                    cv2.addWeighted(overlay, 0, np.ones_like(overlay) * (0, 50, 0), 0.05, 0),
                    0, 0, frame)

    return frame


def resize_frame(frame, width=None, height=None):
    """
    Resize frame while maintaining aspect ratio.

    Args:
        frame: Input frame
        width: Target width (optional)
        height: Target height (optional)

    Returns:
        Resized frame
    """
    if width is None and height is None:
        return frame

    h, w = frame.shape[:2]

    if width is not None:
        scale = width / w
        new_w = width
        new_h = int(h * scale)
    else:
        scale = height / h
        new_h = height
        new_w = int(w * scale)

    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def calculate_fps(start_time, frame_count):
    """
    Calculate frames per second.

    Args:
        start_time: Start time in seconds
        frame_count: Number of frames processed

    Returns:
        FPS value
    """
    import time
    elapsed = time.time() - start_time
    if elapsed == 0:
        return 0
    return frame_count / elapsed


def draw_fps_counter(frame, fps):
    """
    Draw FPS counter on frame.

    Args:
        frame: Video frame
        fps: Current FPS value

    Returns:
        Frame with FPS counter
    """
    color = (0, 255, 0) if fps >= 20 else (0, 165, 255) if fps >= 10 else (0, 0, 255)

    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame


def draw_detector_status(frame, face_detected, phone_detector_loaded):
    """
    Draw detector status indicators.

    Args:
        frame: Video frame
        face_detected: Whether face is detected
        phone_detector_loaded: Whether phone detector is loaded

    Returns:
        Frame with status indicators
    """
    y_offset = frame.shape[0] - 10

    # Face detection status
    face_color = (0, 255, 0) if face_detected else (0, 0, 255)
    face_status = "Face: OK" if face_detected else "Face: --"
    cv2.putText(frame, face_status, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)

    # Phone detector status
    detector_color = (0, 255, 0) if phone_detector_loaded else (0, 0, 255)
    detector_status = "Phone AI: Ready" if phone_detector_loaded else "Phone AI: Off"
    cv2.putText(frame, detector_status, (120, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, detector_color, 1)

    return frame


def safe_get_landmark(landmarks, index, default=(0, 0)):
    """
    Safely get landmark by index with bounds checking.

    Args:
        landmarks: List of landmarks
        index: Index to retrieve
        default: Default value if index out of bounds

    Returns:
        Landmark tuple or default
    """
    if landmarks is None or index < 0 or index >= len(landmarks):
        return default
    return landmarks[index]
