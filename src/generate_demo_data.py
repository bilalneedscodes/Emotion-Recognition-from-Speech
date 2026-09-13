"""
generate_demo_data.py
----------------------
Generates a synthetic MFCC feature dataset so the full training pipeline
can be demonstrated and unit-tested without needing to download a real
speech-emotion dataset (RAVDESS / TESS / EMO-DB require manual download —
see the README).

The synthetic features are drawn from class-dependent Gaussian
distributions over a 40-dim MFCC-like feature vector, which is enough to
exercise the whole pipeline (feature scaling, train/test split, model
training, evaluation) end-to-end. Swap this out for
`extract_features_from_dataset()` in feature_extraction.py once you have
real audio files.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
N_MFCC = 40


def generate_demo_dataset(n_per_class: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for i, emotion in enumerate(EMOTIONS):
        # Give each emotion a distinct mean "signature" across the MFCC
        # coefficients so the classes are separable, similar to how real
        # emotions produce distinguishable spectral patterns.
        class_mean = rng.normal(0, 1, N_MFCC) + i * 0.6
        for _ in range(n_per_class):
            mfcc = class_mean + rng.normal(0, 1.3, N_MFCC)
            row = {f"mfcc_{j+1}": mfcc[j] for j in range(N_MFCC)}
            row["emotion"] = emotion
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


if __name__ == "__main__":
    df = generate_demo_dataset()
    df.to_csv("data/emotion_features_demo.csv", index=False)
    print(f"Saved {len(df)} rows to data/emotion_features_demo.csv")
    print(df["emotion"].value_counts())
