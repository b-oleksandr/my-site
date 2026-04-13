from __future__ import annotations

import numpy as np

from .mediapipe_hand import HandLandmarks


def normalize_landmarks(hand: HandLandmarks) -> np.ndarray:
    """Normalize by wrist position and max distance to make model robust."""
    pts = hand.points.copy()  # shape (21, 3)
    wrist = pts[0]
    pts -= wrist  # translate to wrist-origin
    max_range = np.max(np.linalg.norm(pts[:, :2], axis=1)) or 1.0
    pts[:, :2] /= max_range
    pts[:, 2] /= max_range
    return pts


def landmarks_to_feature_vector(hand: HandLandmarks) -> np.ndarray:
    """Flatten normalized landmarks to 1D feature vector."""
    norm_pts = normalize_landmarks(hand)
    return norm_pts.flatten()

