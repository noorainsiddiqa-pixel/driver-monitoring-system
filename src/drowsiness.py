"""Drowsiness detection using Eye Aspect Ratio (EAR) and yawning detection."""

import cv2
import numpy as np
from collections import deque
from .config import Config, EyeLandmarks


def calculate_eye_aspect_ratio(landmarks, top, bottom, left, right):
    """
    Calculate Eye Aspect Ratio for a single eye.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Where p1-p6 are the eye landmarks in clockwise order.

    Args:
        landmarks: List of facial landmarks (x, y tuples)
        top, bottom, left, right: Landmark indices for the eye

    Returns:
        float: Eye Aspect Ratio value
    """
    # Vertical distances
    v1 = np.linalg.norm(np.array(landmarks[top]) - np.array(landmarks[bottom]))
    v2 = np.linalg.norm(np.array(landmarks[EyeLandmarks.LEFT_EYE_UPPER_LID if top == EyeLandmarks.LEFT_EYE_TOP else EyeLandmarks.RIGHT_EYE_UPPER_LID]) -
                        np.array(landmarks[EyeLandmarks.LEFT_EYE_LOWER_LID if bottom == EyeLandmarks.LEFT_EYE_BOTTOM else EyeLandmarks.RIGHT_EYE_LOWER_LID]))

    # Horizontal distance
    h_dist = np.linalg.norm(np.array(landmarks[left]) - np.array(landmarks[right]))

    if h_dist == 0:
        return 0.0

    ear = (v1 + v2) / (2.0 * h_dist)
    return ear


def detect_drowsiness(frame, face_landmarks, ear_history, frame_count):
    """
    Detect drowsiness based on EAR and frame history.

    Args:
        frame: Current video frame
        face_landmarks: MediaPipe face landmarks
        ear_history: deque of recent EAR values
        frame_count: Current frame counter

    Returns:
        tuple: (is_drowsy, drowsiness_level, updated_ear_history, frame)
    """
    if not face_landmarks:
        return False, 0, ear_history, frame

    landmarks = []
    h, w, _ = frame.shape

    # Extract all landmark points
    for lm in face_landmarks.landmark:
        landmarks.append((int(lm.x * w), int(lm.y * h)))

    # Calculate EAR for both eyes
    left_ear = calculate_eye_aspect_ratio(
        landmarks,
        EyeLandmarks.LEFT_EYE_TOP,
        EyeLandmarks.LEFT_EYE_BOTTOM,
        EyeLandmarks.LEFT_EYE_LEFT,
        EyeLandmarks.LEFT_EYE_RIGHT
    )

    right_ear = calculate_eye_aspect_ratio(
        landmarks,
        EyeLandmarks.RIGHT_EYE_TOP,
        EyeLandmarks.RIGHT_EYE_BOTTOM,
        EyeLandmarks.RIGHT_EYE_LEFT,
        EyeLandmarks.RIGHT_EYE_RIGHT
    )

    # Average EAR
    ear = (left_ear + right_ear) / 2.0
    ear_history.append(ear)

    # Keep only last N frames
    if len(ear_history) > Config.DROWSINESS_FRAMES:
        ear_history.popleft()

    # Check for drowsiness
    is_drowsy = False
    drowsiness_level = 0  # 0=normal, 1=warning, 2=drowsy

    if len(ear_history) >= Config.DROWSINESS_FRAMES:
        avg_ear = np.mean(ear_history)

        if avg_ear < Config.EAR_THRESHOLD:
            is_drowsy = True
            # Calculate drowsiness level based on how long eyes are closed
            closed_ratio = sum(1 for e in ear_history if e < Config.EAR_THRESHOLD) / len(ear_history)
            drowsiness_level = 2 if closed_ratio > 0.8 else 1

    # Draw eye landmarks
    _draw_eye_landmarks(frame, landmarks, left_ear, right_ear, ear_history)

    return is_drowsy, drowsiness_level, ear_history, frame


def _draw_eye_landmarks(frame, landmarks, left_ear, right_ear, ear_history):
    """Draw eye landmarks and EAR values on frame."""
    h, w, _ = frame.shape

    # Left eye points
    left_points = [
        landmarks[EyeLandmarks.LEFT_EYE_LEFT],
        landmarks[EyeLandmarks.LEFT_EYE_TOP],
        landmarks[EyeLandmarks.LEFT_EYE_RIGHT],
        landmarks[EyeLandmarks.LEFT_EYE_BOTTOM],
    ]

    # Right eye points
    right_points = [
        landmarks[EyeLandmarks.RIGHT_EYE_LEFT],
        landmarks[EyeLandmarks.RIGHT_EYE_TOP],
        landmarks[EyeLandmarks.RIGHT_EYE_RIGHT],
        landmarks[EyeLandmarks.RIGHT_EYE_BOTTOM],
    ]

    # Draw eye contours
    cv2.polylines(frame, [np.array(left_points)], True, (0, 255, 0), 2)
    cv2.polylines(frame, [np.array(right_points)], True, (0, 255, 0), 2)

    # Display EAR values
    cv2.putText(frame, f"L-EAR: {left_ear:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"R-EAR: {right_ear:.3f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # EAR history graph
    if len(ear_history) > 10:
        graph_x, graph_y = 10, 100
        graph_w, graph_h = 150, 50

        # Draw graph background
        cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (200, 200, 200), -1)

        # Draw threshold line
        threshold_y = graph_y + graph_h - int((Config.EAR_THRESHOLD / 0.5) * graph_h)
        cv2.line(frame, (graph_x, threshold_y), (graph_x + graph_w, threshold_y), (0, 0, 255), 1)

        # Draw EAR history
        for i in range(1, len(ear_history)):
            x1 = graph_x + int((i - 1) / len(ear_history) * graph_w)
            x2 = graph_x + int(i / len(ear_history) * graph_w)
            y1 = graph_y + graph_h - int(min(ear_history[i-1], 0.5) / 0.5 * graph_h)
            y2 = graph_y + graph_h - int(min(ear_history[i], 0.5) / 0.5 * graph_h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


def detect_yawn(frame, face_landmarks):
    """
    Detect yawning based on mouth aspect ratio.

    Args:
        frame: Current video frame
        face_landmarks: MediaPipe face landmarks

    Returns:
        tuple: (is_yawning, mouth_ratio, frame)
    """
    if not face_landmarks:
        return False, 0.0, frame

    h, w, _ = frame.shape
    landmarks = []

    for lm in face_landmarks.landmark:
        landmarks.append((int(lm.x * w), int(lm.y * h)))

    # Mouth vertical and horizontal distances
    mouth_v = np.linalg.norm(np.array(landmarks[EyeLandmarks.MOUTH_TOP]) -
                             np.array(landmarks[EyeLandmarks.MOUTH_BOTTOM]))
    mouth_h = np.linalg.norm(np.array(landmarks[EyeLandmarks.MOUTH_LEFT]) -
                             np.array(landmarks[EyeLandmarks.MOUTH_RIGHT]))

    if mouth_h == 0:
        return False, 0.0, frame

    mouth_ratio = mouth_v / mouth_h

    # Yawn threshold (mouth open wide)
    is_yawning = mouth_ratio > 0.5

    # Draw mouth landmarks
    mouth_points = [
        landmarks[EyeLandmarks.MOUTH_LEFT],
        landmarks[EyeLandmarks.MOUTH_TOP],
        landmarks[EyeLandmarks.MOUTH_RIGHT],
        landmarks[EyeLandmarks.MOUTH_BOTTOM],
    ]
    cv2.polylines(frame, [np.array(mouth_points)], True, (255, 0, 0), 2)

    return is_yawning, mouth_ratio, frame
