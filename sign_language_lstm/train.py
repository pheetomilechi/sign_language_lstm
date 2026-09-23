"""
train.py
--------
Trains the LSTM sign-language classifier on the preprocessed dataset produced
by preprocess.py, with TensorBoard logging, early stopping, and checkpointing
of the best model.

Usage:
    python train.py
"""

import os
import sys

if __package__ in (None, ""):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from sign_language_lstm.config import (
        ACTIONS, PROCESSED_PATH, MODEL_DIR, MODEL_PATH, LOG_DIR,
        EPOCHS, BATCH_SIZE,
    )
    from sign_language_lstm.model import build_model
else:
    from .config import (
        ACTIONS, PROCESSED_PATH, MODEL_DIR, MODEL_PATH, LOG_DIR,
        EPOCHS, BATCH_SIZE,
    )
    from .model import build_model

import numpy as np
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def load_processed():
    X_train = np.load(os.path.join(PROCESSED_PATH, "X_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_PATH, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_PATH, "y_train.npy"))
    y_test = np.load(os.path.join(PROCESSED_PATH, "y_test.npy"))
    return X_train, X_test, y_train, y_test


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_processed()
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    model = build_model(num_classes=len(ACTIONS))
    model.summary()

    callbacks = [
        TensorBoard(log_dir=LOG_DIR),
        EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor="val_categorical_accuracy",
                         save_best_only=True, mode="max"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ModelCheckpoint already saved the best weights, but ensure a final save too.
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    final_val_acc = max(history.history.get("val_categorical_accuracy", [0]))
    print(f"Best validation accuracy: {final_val_acc:.4f}")


if __name__ == "__main__":
    main()
