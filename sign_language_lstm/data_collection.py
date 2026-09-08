"""
data_collection.py
-------------------
Records labeled keypoint sequences from a webcam for every sign in config.ACTIONS.

For each action, it captures NO_SEQUENCES separate "takes", each SEQUENCE_LENGTH
frames long. Every frame's extracted keypoints (see mp_utils.extract_keypoints)
are saved as an individual .npy file:

    data/keypoints/<action>/<sequence_index>/<frame_index>.npy

This folder layout mirrors the classic MediaPipe-Holistic + LSTM action-recognition
pipeline and is what preprocess.py expects when it builds the training arrays.

Controls while running:
    - The script automatically cycles through actions and sequences.
    - Press 'q' at any time to abort early.
    - A 2-second "Get Ready" pause is shown at the start of every new sequence
      so you have time to reset your hands to a neutral position.
"""

import os
import cv2
import numpy as np

from config import ACTIONS, DATA_PATH, NO_SEQUENCES, SEQUENCE_LENGTH
from mp_utils import mediapipe_detection, draw_landmarks, extract_keypoints, new_holistic_model


def make_folders():
    for action in ACTIONS:
        for sequence in range(NO_SEQUENCES):
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)


def collect():
    make_folders()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check your camera connection/permissions.")

    with new_holistic_model(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in ACTIONS:
            for sequence in range(NO_SEQUENCES):
                for frame_num in range(SEQUENCE_LENGTH):
                    ok, frame = cap.read()
                    if not ok:
                        print("Warning: failed to read frame from webcam.")
                        continue

                    frame, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(frame, results)

                    if frame_num == 0:
                        cv2.putText(frame, "GET READY", (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                        cv2.putText(frame, f'Collecting "{action}" | Take #{sequence}', (15, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.imshow("Data Collection", frame)
                        cv2.waitKey(1500)
                    else:
                        cv2.putText(frame, f'Collecting "{action}" | Take #{sequence} | Frame {frame_num}',
                                    (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.imshow("Data Collection", frame)

                    keypoints = extract_keypoints(results)
                    out_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                    np.save(out_path, keypoints)

                    if cv2.waitKey(10) & 0xFF == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        print("Aborted by user.")
                        return

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Collected {len(ACTIONS)} actions x {NO_SEQUENCES} sequences "
          f"x {SEQUENCE_LENGTH} frames into {DATA_PATH}")


if __name__ == "__main__":
    collect()
