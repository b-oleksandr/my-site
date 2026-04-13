from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import sys

# Ensure the top-level src directory (which contains packages like camera, tracker, etc.)
# is on sys.path, regardless of where this script is executed from.
SRC_CANDIDATE = Path(__file__).resolve().parent
if not (SRC_CANDIDATE / "camera").is_dir():
    SRC_CANDIDATE = SRC_CANDIDATE.parent
if str(SRC_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(SRC_CANDIDATE))

from gesture_model import train_and_save


def main():
    parser = argparse.ArgumentParser(description="Train gesture classifier.")
    parser.add_argument("--csv", type=Path, default=Path(SRC_CANDIDATE.parent / "data/processed/landmarks.csv"))
    parser.add_argument("--model_out", type=Path, default=Path(SRC_CANDIDATE.parent / "models/gesture_model.joblib"))
    parser.add_argument("--labels_out", type=Path, default=Path(SRC_CANDIDATE.parent / "models/labels.json"))
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV не знайдено: {args.csv}. Спершу запустіть collect_data.")

    df = pd.read_csv(args.csv)
    y = df["label"].values
    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = train_and_save(X_train, y_train, args.model_out, args.labels_out)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    print("Звіт точності:")
    print(report)


if __name__ == "__main__":
    main()

