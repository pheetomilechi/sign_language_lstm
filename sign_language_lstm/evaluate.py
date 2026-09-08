"""
evaluate.py
-----------
Loads the trained model and held-out test set, then reports:
    - overall categorical accuracy
    - a per-class classification report (precision/recall/F1)
    - a confusion matrix, plotted with matplotlib/seaborn and saved to disk

Usage:
    python evaluate.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model

from config import ACTIONS, PROCESSED_PATH, MODEL_PATH, MODEL_DIR


def main():
    X_test = np.load(os.path.join(PROCESSED_PATH, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_PATH, "y_test.npy"))

    model = load_model(MODEL_PATH)

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)
    print(f"Test accuracy: {acc:.4f}\n")

    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=list(ACTIONS)))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=ACTIONS, yticklabels=ACTIONS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Sign Language LSTM")
    plt.tight_layout()

    out_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    print(f"Confusion matrix saved to {out_path}")


if __name__ == "__main__":
    main()
