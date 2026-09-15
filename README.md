# CodeAlpha_EmotionRecognitionFromSpeech

**Task:** Recognize human emotions (happy, sad, angry, etc.) from speech audio.

## 📌 Overview

This project classifies emotions from speech using **MFCC (Mel-Frequency
Cepstral Coefficient)** features extracted from audio, fed into a neural
network (MLP) classifier built with TensorFlow/Keras.

It ships in two parts:

1. **A working demo pipeline** that runs immediately with zero setup, using
   a synthetic MFCC feature dataset so you can see the full train →
   evaluate → save pipeline in action without waiting on a large download.
2. **A real feature-extraction module** (`src/feature_extraction.py`) using
   `librosa`, ready to point at an actual speech-emotion dataset such as
   RAVDESS, TESS, or EMO-DB.

## 🗂 Project Structure

```
CodeAlpha_EmotionRecognitionFromSpeech/
├── src/
│   ├── generate_demo_data.py   # Synthetic MFCC dataset for the demo pipeline
│   ├── feature_extraction.py   # Real MFCC extraction from .wav files (librosa)
│   └── train_model.py          # Trains/evaluates the MLP classifier
├── data/                       # Feature CSVs (generated / extracted)
├── models/                     # Saved model, scaler, label encoder
├── outputs/                    # Training curves, confusion matrix
├── requirements.txt
└── README.md
```

## 🚀 Quick Start (demo, no download needed)

```bash
pip install -r requirements.txt
python src/generate_demo_data.py   # creates data/emotion_features_demo.csv
python src/train_model.py          # trains on the demo data by default
```

## 🎙 Using a Real Speech Dataset (RAVDESS)

1. Download RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song):
   https://zenodo.org/record/1188976
   (TESS: https://tspace.library.utoronto.ca/handle/1807/24487,
   EMO-DB: http://emodb.bilderbar.info/start.html)
2. Extract it into `data/RAVDESS/` (keep the nested Actor_XX folders).
3. Extract MFCC features:
   ```bash
   python src/feature_extraction.py --data_dir data/RAVDESS --out data/emotion_features.csv
   ```
4. Train on the real features:
   ```bash
   python src/train_model.py --data_path data/emotion_features.csv
   ```

`feature_extraction.py` includes a ready-made filename parser for RAVDESS's
naming convention (`03-01-06-...`). For TESS or EMO-DB, add a similar
parser function — their filenames encode emotion differently.

## 🧠 Model

A fully-connected neural network (MLP):
`Dense(256) → BatchNorm → Dropout → Dense(128) → BatchNorm → Dropout → Dense(64) → Softmax`,
trained with early stopping on validation loss. This is a strong, fast
baseline for MFCC feature vectors — the README's "Extending" section below
covers upgrading to CNN/LSTM on raw spectrograms for higher accuracy on
real audio.

## 🔧 Extending This Project

- Swap the MLP for a **CNN or LSTM** operating directly on MFCC
  *sequences* (not just the time-averaged vector) for better accuracy on
  real speech.
- Add data augmentation (pitch shift, time stretch, noise injection) —
  common for small emotion datasets like RAVDESS (~1,440 clips).
- Combine RAVDESS + TESS + EMO-DB for a larger, more robust training set.

## 🛠 Tech Stack

Python, librosa, NumPy, pandas, scikit-learn, TensorFlow/Keras, matplotlib, seaborn

## ✍️ Author

Built as part of the CodeAlpha Machine Learning Internship.
