# Continuous Sign Language Recognition using LSTM

A complete, working ML/DL pipeline that recognizes sign-language words **continuously**
from a live webcam feed, using **MediaPipe Holistic** for keypoint extraction and a
stacked **LSTM** network for temporal sequence classification.

Unlike isolated sign-language recognition (classifying one pre-cut clip at a time),
this project performs *continuous* recognition: it watches an ongoing, unsegmented
video stream and figures out, frame by frame, when a sign has been performed —
building up a running sentence as you sign.

---

## How it works

```
Webcam frame
      │
      ▼
MediaPipe Holistic  ──►  Pose (33 pts) + Left hand (21 pts) + Right hand (21 pts)
      │                        flattened into one feature vector per frame
      ▼
Rolling window of the last 30 frames  (a sliding window, updated every new frame)
      │
      ▼
3-layer LSTM classifier  ──►  softmax probability over known signs
      │
      ▼
Confidence threshold + stability voting + cooldown/dedup
      │
      ▼
Confirmed word appended to the running output sentence
```

### Why LSTM?
Sign language is inherently temporal — the same hand shape means different things
depending on the motion trajectory before and after it. An LSTM processes the
sequence of keypoints frame-by-frame and retains a memory of past motion, making it
well suited to distinguishing signs that share static hand shapes but differ in
movement.

### Making it *continuous*
Three mechanisms turn a plain "classify this clip" model into a live continuous
recognizer (implemented in `real_time_prediction.py`):

1. **Rolling window** — a `deque` always holds the most recent 30 frames of
   keypoints. The model re-predicts on every new frame, not just at clip
   boundaries, so there's no need to manually segment the video into per-sign
   clips.
2. **Stability voting** — the last 10 per-frame predictions are pooled, and a
   sign is only "confirmed" once it wins a majority. This smooths out single-frame
   flicker from motion blur or transitional hand shapes.
3. **Cooldown / de-duplication** — once a sign is confirmed, it won't be appended
   again immediately, preventing one held sign from spamming the output sentence
   with duplicates.

---

## Project structure

```
sign_language_lstm/
├── config.py                 # all tunable constants (vocabulary, sequence length, thresholds...)
├── mp_utils.py                # MediaPipe keypoint extraction & drawing helpers (shared everywhere)
├── data_collection.py         # records labeled keypoint sequences from your webcam
├── preprocess.py              # builds train/test numpy arrays + label map from raw recordings
├── model.py                   # LSTM architecture definition
├── train.py                   # trains the model, with early stopping + checkpointing
├── evaluate.py                # test accuracy, classification report, confusion matrix
├── real_time_prediction.py    # CONTINUOUS live recognition (the main deliverable)
├── requirements.txt
├── data/
│   ├── keypoints/              # raw per-frame .npy files, organized by action/sequence/frame
│   └── processed/              # X_train/X_test/y_train/y_test .npy arrays
├── models/                     # saved model (.h5), label_map.json, confusion_matrix.png
└── logs/                       # TensorBoard logs
```

---

## Setup

Use Python 3.11 for this project. TensorFlow 2.13-2.15 does not provide
Windows packages for Python 3.13 or 3.14.

```bash
# Run these commands from the outer sign_language_lstm folder, the one that
# contains the inner sign_language_lstm source folder.
py -3.11 -m venv venv
source venv/Scripts/activate    # Git Bash on Windows
# .\venv\Scripts\Activate.ps1  # PowerShell
pip install -r requirements.txt
```

Requires a webcam. Tested against Python 3.11, TensorFlow 2.13–2.15, and
MediaPipe 0.10.x.

---

## Usage

### 1. Define your vocabulary
Edit `ACTIONS` in `config.py` — this is the list of signs/words the model will learn.
Defaults to a small starter set: `hello, thanks, please, sorry, yes, no, help, name, good, bye`.

### 2. Collect training data
```bash
python data_collection.py
```
For each action, this records `NO_SEQUENCES` (default 40) separate takes of
`SEQUENCE_LENGTH` (default 30) frames each. Perform the sign naturally and repeat
with slight variation (position, speed, angle) across takes — this variety is what
lets the LSTM generalize instead of memorizing one exact recording.
Press `q` to abort early if needed.

> Tip: aim for at least 40–60 takes per sign for a reasonably robust small-vocabulary
> model. More signs and more takes = better accuracy, but longer collection time.

### 3. Preprocess into train/test arrays
```bash
python preprocess.py
```
Builds `data/processed/{X_train,X_test,y_train,y_test}.npy` and `models/label_map.json`.

### 4. Train the LSTM
```bash
python train.py
```
Trains with early stopping (patience 25 on validation loss) and saves the best
checkpoint to `models/sign_lstm_model.h5`. Monitor training live with:
```bash
tensorboard --logdir logs
```

### 5. Evaluate
```bash
python evaluate.py
```
Prints test accuracy + a per-class precision/recall/F1 report, and saves a
confusion-matrix heatmap to `models/confusion_matrix.png`.

### 6. Run continuous, real-time recognition
```bash
python real_time_prediction.py
```
Opens your webcam and starts building a live sentence from recognized signs on
screen. Press `c` to clear the current sentence, `q` to quit.

### 7. Run the browser-based demo
```bash
streamlit run sign_language_lstm/streamlit_app.py
```
Open the URL shown by Streamlit in your browser, allow camera access, and start
the webcam. The live video includes the recognized sentence and the page provides
buttons to clear the sentence or pause/resume recording. This uses
`streamlit-webrtc`, so camera access is handled by the browser instead of an
OpenCV window.

---

## Tuning continuous recognition

All of these live in `config.py`:

| Parameter | Effect |
|---|---|
| `PREDICTION_THRESHOLD` | Higher = fewer false positives, but may miss quick/subtle signs. |
| `STABILITY_WINDOW` | Larger = smoother but slightly more laggy confirmation. |
| `COOLDOWN_FRAMES` | Larger = less risk of duplicate words when a sign is held, but slower to register a deliberately repeated sign. |
| `SEQUENCE_LENGTH` | Frames per window; increase for longer/slower signs, decrease for snappier response. |

---

## Extending this project

- **More signs / bigger vocabulary**: just add entries to `ACTIONS` and re-run data
  collection + training. The architecture scales to hundreds of classes with enough
  data.
- **Sentence-level grammar**: add a language model or simple n-gram smoothing on top
  of the emitted word sequence to correct likely mis-recognitions in context.
- **Multi-person robustness**: collect data from multiple signers to reduce
  overfitting to one person's signing style.
- **Bidirectional LSTM / Transformer**: swap `model.py`'s architecture for a
  `Bidirectional(LSTM(...))` stack or a small Transformer encoder for potentially
  higher accuracy, at the cost of needing more data and losing pure real-time
  causality (bidirectional models need to see into the "future" of the window).
- **Deployment**: export with `model.save()` (already `.h5`) and convert to
  TensorFlow Lite for mobile/edge deployment if needed.

---

## Notes on real-world accuracy

This is an educational/prototype-grade pipeline. A single-signer, ~10-word
vocabulary setup like the default config can realistically reach 90%+ validation
accuracy with clean data. For production-grade continuous sign language translation
(large vocabularies, multiple signers, co-articulation between signs), you would
typically need: much larger datasets (e.g. WLASL, How2Sign, PHOENIX-2014T), stronger
architectures (Transformers, Graph Convolutional Networks over hand/body joints),
and dedicated continuous-sign-language-recognition techniques such as CTC
(Connectionist Temporal Classification) loss to handle unsegmented sequences
without needing hand-picked thresholds like the ones used here.
