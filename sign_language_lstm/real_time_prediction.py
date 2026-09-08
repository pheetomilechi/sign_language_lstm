"""
real_time_prediction.py
------------------------
Runs CONTINUOUS sign language recognition on a live webcam feed.

Unlike isolated sign recognition (one classification per pre-segmented clip),
continuous recognition has to decide, frame by frame, whether a new sign has
just been completed within an unsegmented, ongoing video stream. This script
implements that with a rolling-window + stability-voting approach:

  1. ROLLING WINDOW
     Keep the most recent SEQUENCE_LENGTH frames of keypoints in a deque.
     Once the window is full, feed it to the LSTM on every new frame to get a
     fresh probability distribution over ACTIONS. This means predictions are
     made continuously, not just at clip boundaries.

  2. CONFIDENCE THRESHOLDING
     Only consider a prediction if its softmax probability exceeds
     PREDICTION_THRESHOLD. This filters out low-confidence noise, e.g. during
     the transition between two signs.

  3. STABILITY VOTING
     Keep a short history (STABILITY_WINDOW) of the most recent per-frame
     top predictions. A sign is only "confirmed" once it is the majority
     vote across that history. This smooths out single-frame flicker caused
     by motion blur or brief misclassification.

  4. COOLDOWN / DEDUPLICATION
     Once a sign is confirmed and appended to the output sentence, the same
     sign is not appended again until either a different sign is confirmed or
     COOLDOWN_FRAMES have elapsed. This stops one held sign from spamming the
     sentence buffer with duplicate words.

The result is a running "sentence" of recognized signs displayed on screen,
which is the practical behavior expected of a continuous sign-language
recognizer (as opposed to single-word/single-clip classifiers).
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from config import MODEL_PATH
from mp_utils import new_holistic_model
from recognizer import ContinuousRecognizer, ensure_model_files_exist, load_label_map_inverse


def run():
    ensure_model_files_exist()

    model = load_model(MODEL_PATH)
    recognizer = ContinuousRecognizer(model, load_label_map_inverse())

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check your camera connection/permissions.")

    with new_holistic_model(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = recognizer.process_frame(frame, holistic)

            # --- UI overlay ---
            overlay_h = 60
            cv2.rectangle(
                frame, (0, 0), (frame.shape[1], overlay_h), (30, 30, 30), -1)
            cv2.putText(frame, " ".join(recognizer.sentence), (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

            if recognizer.last_word:
                label = f"{recognizer.last_word} ({recognizer.last_confidence:.2f})"
                cv2.putText(frame, label, (10, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(
                "Continuous Sign Language Recognition (press 'q' to quit, 'c' to clear)", frame)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                recognizer.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
