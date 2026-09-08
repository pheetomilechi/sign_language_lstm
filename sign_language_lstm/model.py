"""
model.py
--------
Defines the LSTM-based sequence classifier used to recognize signs from
sequences of MediaPipe keypoints.

Architecture rationale:
    - Three stacked LSTM layers progressively compress the temporal sequence
      of hand/pose movements into a fixed-size representation.
    - Dropout layers combat overfitting, since sign-language datasets
      collected by a single person are typically small.
    - Dense layers at the end act as the classifier head over the learned
      temporal features.
    - Softmax output gives a probability distribution over ACTIONS, which is
      exactly what the continuous-prediction sliding-window logic needs.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam

from config import SEQUENCE_LENGTH, FEATURE_LENGTH, LEARNING_RATE


def build_model(num_classes: int) -> Sequential:
    model = Sequential(name="sign_language_lstm")

    # Masking lets the network ignore zero-padded frames (e.g. from a hand
    # briefly leaving the frame) instead of treating them as real signal.
    model.add(Masking(mask_value=0.0, input_shape=(SEQUENCE_LENGTH, FEATURE_LENGTH)))

    model.add(LSTM(64, return_sequences=True, activation="tanh"))
    model.add(Dropout(0.3))

    model.add(LSTM(128, return_sequences=True, activation="tanh"))
    model.add(Dropout(0.3))

    model.add(LSTM(64, return_sequences=False, activation="tanh"))
    model.add(Dropout(0.3))

    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model


if __name__ == "__main__":
    # Quick sanity check: prints the architecture summary.
    from config import ACTIONS
    m = build_model(num_classes=len(ACTIONS))
    m.summary()
