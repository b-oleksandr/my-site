from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


@dataclass
class HandLandmarks:
    points: np.ndarray  # shape: (21, 3) with x, y, z normalized to image size


class HandTracker:
    """Detects hands and returns landmarks + helper drawing utilities."""

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.6,
    ):
        self.hands = mp_hands.Hands(
            model_complexity=1,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> List[HandLandmarks]:
        """Returns list of detected hand landmarks."""
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_shape = frame_bgr.shape[:2]

        hands: List[HandLandmarks] = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                coords = np.array(
                    [
                        [
                            lm.x * image_shape[1],
                            lm.y * image_shape[0],
                            lm.z * image_shape[1],
                        ]
                        for lm in hand_landmarks.landmark
                    ],
                    dtype=np.float32,
                )
                hands.append(HandLandmarks(points=coords))
        return hands

    def draw(self, frame_bgr: np.ndarray, hand_landmarks_list: Sequence[HandLandmarks]):
        for hand in hand_landmarks_list:
            mp_drawing.draw_landmarks(
                frame_bgr,
                self._to_mediapipe_landmarks(hand),
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

    @staticmethod
    def _to_mediapipe_landmarks(hand: HandLandmarks):
        """Convert numpy landmarks back to MediaPipe format for drawing."""
        from mediapipe.framework.formats import landmark_pb2

        landmark_list = []
        for x, y, z in hand.points:
            landmark_list.append(landmark_pb2.NormalizedLandmark(x=x, y=y, z=z))
        return landmark_pb2.NormalizedLandmarkList(landmark=landmark_list)

