"""Reusable continuous sign-language recognition state machine."""

import json
import os
from collections import Counter, deque
from threading import RLock

import numpy as np

from config import (
    COOLDOWN_FRAMES,
    LABEL_MAP_PATH,
    MAX_SENTENCE_LENGTH,
    MODEL_PATH,
    PREDICTION_THRESHOLD,
    SEQUENCE_LENGTH,
    STABILITY_WINDOW,
)
from mp_utils import draw_landmarks, extract_keypoints, mediapipe_detection


def load_label_map_inverse():
    with open(LABEL_MAP_PATH, "r") as label_file:
        label_map = json.load(label_file)
    return {int(index): label for label, index in label_map.items()}


class ContinuousRecognizer:
    """Run the rolling-window, voting, and cooldown logic on video frames."""

    def __init__(self, model, inverse_label_map):
        self.model = model
        self.inverse_label_map = inverse_label_map
        self._lock = RLock()
        self.reset()
        self.recording = True

    @property
    def sentence(self):
        with self._lock:
            return list(self._sentence)

    def set_recording(self, recording):
        with self._lock:
            self.recording = recording

    def reset(self):
        with self._lock:
            self._frame_window = deque(maxlen=SEQUENCE_LENGTH)
            self._vote_history = deque(maxlen=STABILITY_WINDOW)
            self._sentence = []
            self._last_confirmed_word = None
            self._cooldown_counter = 0
            self.last_word = None
            self.last_confidence = 0.0

    def process_frame(self, frame, holistic):
        """Process one BGR frame and return it with MediaPipe landmarks drawn."""
        frame, results = mediapipe_detection(frame, holistic)
        draw_landmarks(frame, results)

        with self._lock:
            self.last_word = None
            self.last_confidence = 0.0

            if not self.recording:
                return frame

            self._frame_window.append(extract_keypoints(results))
            if len(self._frame_window) < SEQUENCE_LENGTH:
                return frame

            input_seq = np.expand_dims(np.array(self._frame_window), axis=0)
            probabilities = self.model.predict(input_seq, verbose=0)[0]
            top_index = int(np.argmax(probabilities))
            top_confidence = float(probabilities[top_index])

            if top_confidence > PREDICTION_THRESHOLD:
                self._vote_history.append(top_index)
            else:
                self._vote_history.append(-1)

            if self._cooldown_counter > 0:
                self._cooldown_counter -= 1

            if len(self._vote_history) < STABILITY_WINDOW:
                return frame

            majority_index, majority_count = Counter(
                self._vote_history).most_common(1)[0]
            if majority_index == -1 or majority_count < (STABILITY_WINDOW // 2 + 1):
                return frame

            word = self.inverse_label_map[majority_index]
            self.last_word = word
            self.last_confidence = top_confidence
            can_confirm = (
                word != self._last_confirmed_word or self._cooldown_counter == 0
            )
            if can_confirm:
                self._sentence = (
                    self._sentence + [word])[-MAX_SENTENCE_LENGTH:]
                self._last_confirmed_word = word
                self._cooldown_counter = COOLDOWN_FRAMES
                self._vote_history.clear()

        return frame


def ensure_model_files_exist():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run preprocess.py and train.py first."
        )
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(
            f"No label map found at {LABEL_MAP_PATH}. Run preprocess.py first."
        )
