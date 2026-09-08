"""
config.py
---------
Central configuration for the Continuous Sign Language Recognition project.
Edit ACTIONS to match the signs/words you want to recognize.
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "keypoints")          # raw collected keypoints
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")     # X.npy / y.npy for training
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

MODEL_PATH = os.path.join(MODEL_DIR, "sign_lstm_model.h5")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
# The set of signs (words) the model will learn to recognize.
# Add / remove words here. More words per recording session = more classes.
ACTIONS = np.array([
    "hello",
    "thanks",
    "please",
    "sorry",
    "yes",
    "no",
    "help",
    "name",
    "good",
    "bye",
])

# ---------------------------------------------------------------------------
# Data collection / sequence parameters
# ---------------------------------------------------------------------------
NO_SEQUENCES = 40          # number of video samples ("takes") to record per sign
SEQUENCE_LENGTH = 30       # number of frames per sequence (i.e. per sign sample)

# Keypoint vector size produced by MediaPipe Holistic:
#   Pose:        33 landmarks * 4 values (x, y, z, visibility) = 132
#   Face mesh:  468 landmarks * 3 values (x, y, z)             = 1404
#   Left hand:   21 landmarks * 3 values (x, y, z)             = 63
#   Right hand:  21 landmarks * 3 values (x, y, z)             = 63
# We drop the dense face mesh (1404 values) to keep the feature vector compact
# and focused on the signal that actually matters for sign language: pose + hands.
USE_FACE_MESH = False
POSE_FEATURES = 33 * 4
FACE_FEATURES = 468 * 3
HAND_FEATURES = 21 * 3

FEATURE_LENGTH = POSE_FEATURES + (FACE_FEATURES if USE_FACE_MESH else 0) + HAND_FEATURES * 2

# ---------------------------------------------------------------------------
# Training parameters
# ---------------------------------------------------------------------------
TEST_SIZE = 0.15
RANDOM_STATE = 42
EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

# ---------------------------------------------------------------------------
# Continuous / real-time prediction parameters
# ---------------------------------------------------------------------------
# Continuous recognition works with a rolling window of the last SEQUENCE_LENGTH
# frames, re-predicting every new frame. A prediction is only "confirmed" (i.e.
# appended to the output sentence) when:
#   1. Its confidence exceeds PREDICTION_THRESHOLD, AND
#   2. It is the majority prediction across the last STABILITY_WINDOW predictions
#      (this prevents flickering / single-frame noise from producing spurious words), AND
#   3. It differs from the last confirmed word (to avoid duplicate emission while
#      a sign is being held), unless COOLDOWN_FRAMES have passed.
PREDICTION_THRESHOLD = 0.80
STABILITY_WINDOW = 10
COOLDOWN_FRAMES = 20
MAX_SENTENCE_LENGTH = 8   # max number of words kept on screen / in the output buffer
