from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class GestureModel:
    """Wrapper for loading and using an sklearn model."""

    def __init__(
        self,
        model_path: str | Path = "models/gesture_model.joblib",
        labels_path: str | Path = "models/labels.json",
    ):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.model: Optional[RandomForestClassifier] = None
        self.labels: list[str] = []
        self._load()

    def _load(self):
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
        if self.labels_path.exists():
            with open(self.labels_path, "r", encoding="utf-8") as f:
                self.labels = json.load(f)

    @property
    def ready(self) -> bool:
        return self.model is not None and len(self.labels) > 0

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        if not self.ready:
            return "unknown", 0.0
        features = features.reshape(1, -1)
        proba = self.model.predict_proba(features)[0]
        idx = int(np.argmax(proba))
        label = self.labels[idx]
        confidence = float(proba[idx])
        return label, confidence


def train_and_save(
    X: np.ndarray,
    y: np.ndarray,
    model_out: str | Path,
    labels_out: str | Path,
    n_estimators: int = 200,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X, y)
    labels = sorted(list(set(y)))

    joblib.dump(model, model_out)
    with open(labels_out, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    return model

