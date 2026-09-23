"""
preprocess.py
-------------
Loads the raw per-frame .npy keypoint files produced by data_collection.py,
assembles them into fixed-length sequences per sample, one-hot encodes the
labels, performs a train/test split, and saves the resulting arrays plus the
label map to disk for train.py / evaluate.py to consume.

Output:
    data/processed/X_train.npy, X_test.npy, y_train.npy, y_test.npy
    models/label_map.json
"""

import os
import sys
import json

if __package__ in (None, ""):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from sign_language_lstm.config import (
        ACTIONS, DATA_PATH, PROCESSED_PATH, NO_SEQUENCES, SEQUENCE_LENGTH,
        TEST_SIZE, RANDOM_STATE, LABEL_MAP_PATH, MODEL_DIR,
    )
else:
    from .config import (
        ACTIONS, DATA_PATH, PROCESSED_PATH, NO_SEQUENCES, SEQUENCE_LENGTH,
        TEST_SIZE, RANDOM_STATE, LABEL_MAP_PATH, MODEL_DIR,
    )

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def load_dataset():
    label_map = {label: idx for idx, label in enumerate(ACTIONS)}

    sequences, labels = [], []
    missing = 0

    for action in ACTIONS:
        for sequence in range(NO_SEQUENCES):
            window = []
            sequence_dir = os.path.join(DATA_PATH, action, str(sequence))
            if not os.path.isdir(sequence_dir):
                missing += 1
                continue
            for frame_num in range(SEQUENCE_LENGTH):
                frame_path = os.path.join(sequence_dir, f"{frame_num}.npy")
                if os.path.exists(frame_path):
                    window.append(np.load(frame_path))
                else:
                    # Zero-pad any missing frame so all sequences stay the same length.
                    window.append(np.zeros_like(sequences[0][0]) if sequences else np.zeros(1))
            sequences.append(window)
            labels.append(label_map[action])

    if missing:
        print(f"Warning: {missing} sequence folders were missing and skipped.")

    X = np.array(sequences)
    y = to_categorical(labels, num_classes=len(ACTIONS)).astype(int)
    return X, y, label_map


def main():
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y, label_map = load_dataset()
    print(f"Loaded dataset: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    np.save(os.path.join(PROCESSED_PATH, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_PATH, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_PATH, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_PATH, "y_test.npy"), y_test)

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"Saved processed arrays to {PROCESSED_PATH}")
    print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"Label map saved to {LABEL_MAP_PATH}")


if __name__ == "__main__":
    main()
