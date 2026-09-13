"""
feature_extraction.py
-----------------------
Real MFCC feature extraction for speech-emotion recognition, meant to be
used once you have a real dataset (RAVDESS, TESS, or EMO-DB) downloaded
locally. See the README for download links and the expected folder layout.

Usage:
    python src/feature_extraction.py --data_dir data/RAVDESS --out data/emotion_features.csv

Each dataset has its own filename convention for encoding the emotion
label, so `parse_ravdess_filename` is provided out of the box; add a
similar parser for TESS/EMO-DB if you use those instead.
"""

import argparse
import glob
import os

import librosa
import numpy as np
import pandas as pd

N_MFCC = 40

RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def parse_ravdess_filename(filepath: str) -> str:
    """RAVDESS filenames look like: 03-01-06-01-02-01-12.wav
    The 3rd field (index 2) is the emotion code."""
    fname = os.path.basename(filepath)
    parts = fname.split("-")
    code = parts[2]
    return RAVDESS_EMOTION_MAP.get(code, "unknown")


def extract_mfcc_features(filepath: str, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Load an audio file and return the mean MFCC vector across time."""
    y, sr = librosa.load(filepath, sr=None, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)


def build_dataset(data_dir: str, label_parser=parse_ravdess_filename) -> pd.DataFrame:
    audio_files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    if not audio_files:
        raise FileNotFoundError(
            f"No .wav files found under {data_dir}. Download RAVDESS/TESS/EMO-DB "
            "and point --data_dir at it (see README for links)."
        )

    rows = []
    for filepath in audio_files:
        try:
            features = extract_mfcc_features(filepath)
            label = label_parser(filepath)
            row = {f"mfcc_{i+1}": features[i] for i in range(len(features))}
            row["emotion"] = label
            rows.append(row)
        except Exception as e:
            print(f"Skipping {filepath}: {e}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to folder of .wav files")
    parser.add_argument("--out", default="data/emotion_features.csv")
    args = parser.parse_args()

    df = build_dataset(args.data_dir)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} extracted samples to {args.out}")
