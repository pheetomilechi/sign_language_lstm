# Sign Language LSTM

A continuous sign-language recognition project that uses MediaPipe Holistic keypoints and an LSTM model to detect and interpret sign words from a live webcam stream.

## Overview

This project recognizes sign-language words in real time from a webcam feed. It extracts landmark data from the face, hands, and pose, then uses a temporal LSTM model to classify the current motion sequence.

The project includes:
- data collection for labeled sign sequences
- preprocessing and feature extraction
- model training and evaluation
- continuous real-time prediction from webcam input
- a browser-based Streamlit demo

## Project structure

```text
.
├── README.md
├── requirements.txt
├── runtime.txt
├── packages.txt
├── sign_language_lstm/
│   ├── config.py
│   ├── data_collection.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── recognizer.py
│   ├── real_time_prediction.py
│   ├── streamlit_app.py
│   ├── mp_utils.py
│   ├── convert_to_tflite.py
│   └── README.md
├── .gitignore
└── .devcontainer/
```

## Requirements

- Python 3.11
- TensorFlow 2.13–2.15
- OpenCV
- MediaPipe
- Streamlit
- NumPy

## Setup

```bash
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

### Collect training data

```bash
python sign_language_lstm/data_collection.py
```

### Preprocess data

```bash
python sign_language_lstm/preprocess.py
```

### Train the model

```bash
python sign_language_lstm/train.py
```

### Evaluate the model

```bash
python sign_language_lstm/evaluate.py
```

### Run real-time recognition

```bash
python sign_language_lstm/real_time_prediction.py
```

### Run the Streamlit demo

```bash
streamlit run sign_language_lstm/streamlit_app.py
```

## Notes

This project is intended for experimentation and learning. For production systems, larger datasets and stronger architectures are typically required.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
