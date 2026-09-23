"""
mp_utils.py
-----------
Shared helpers built on MediaPipe Holistic for:
  - running detection on a video frame
  - drawing landmarks on the frame (for visual feedback)
  - flattening detected landmarks into a single feature vector consumed by the LSTM

Keeping this logic in one place ensures data collection, training, and real-time
inference all extract features in exactly the same way.
"""

import os
import sys

if __package__ in (None, ""):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from sign_language_lstm.config import USE_FACE_MESH
else:
    from .config import USE_FACE_MESH

import numpy as np
import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def mediapipe_detection(frame, model):
    """Run a MediaPipe Holistic model on a single BGR frame.

    Returns the (unchanged) frame and the MediaPipe results object.
    """
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return frame, results


def draw_landmarks(frame, results):
    """Draw pose + hand (+ optional face) landmarks on the frame in-place."""
    if results.face_landmarks and USE_FACE_MESH:
        mp_drawing.draw_landmarks(
            frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )
    mp_drawing.draw_landmarks(
        frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )
    mp_drawing.draw_landmarks(
        frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
    )
    mp_drawing.draw_landmarks(
        frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
    )
    return frame


def extract_keypoints(results):
    """Flatten MediaPipe results into a single 1-D numpy feature vector.

    Missing landmarks (e.g. a hand that's out of frame) are zero-filled so the
    feature vector always has a fixed, consistent length (see config.FEATURE_LENGTH).
    """
    pose = (
        np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(33 * 4)
    )
    lh = (
        np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(21 * 3)
    )
    rh = (
        np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(21 * 3)
    )

    if USE_FACE_MESH:
        face = (
            np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten()
            if results.face_landmarks else np.zeros(468 * 3)
        )
        return np.concatenate([pose, face, lh, rh])

    return np.concatenate([pose, lh, rh])


def new_holistic_model(min_detection_confidence=0.5, min_tracking_confidence=0.5):
    """Factory for a MediaPipe Holistic model instance with sane defaults."""
    return mp_holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
