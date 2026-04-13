from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import sys

# Ensure the top-level src directory (which contains packages like camera, tracker, etc.)
# is on sys.path, regardless of where this script is executed from.
SRC_CANDIDATE = Path(__file__).resolve().parent
if not (SRC_CANDIDATE / "camera").is_dir():
    SRC_CANDIDATE = SRC_CANDIDATE.parent
if str(SRC_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(SRC_CANDIDATE))

from camera.camera import Camera
from tracker.mediapipe_hand import HandTracker
from tracker.preprocessing import landmarks_to_feature_vector


def collect_images(output_dir: Path, label: str, num_samples: int = 50) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    camera = Camera()
    if not camera.open():
        raise SystemExit("Не вдалося відкрити камеру.")

    tracker = HandTracker()
    saved_paths: List[Path] = []
    print("Показуйте жест. Натискайте пробіл для збереження кадру, q для виходу.")
    
    target_fps = 5
    frame_time = 1.0 / target_fps  # 0.2 seconds per frame
    last_frame_time = time.time()
    frame = None
    record = False

    while len(saved_paths) < num_samples:
        current_time = time.time()
        elapsed = current_time - last_frame_time

        # Always check for keyboard input to keep UI responsive
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            record = True

        # Only process and display frame if enough time has passed
        if elapsed >= frame_time:

            print(f"[{current_time}; {elapsed:.2f}] Processing frame {len(saved_paths) + 1}/{num_samples}")
            frame = camera.read()
            hands = tracker.process(frame)
            if hands:
                tracker.draw(frame, hands)

            cv2.putText(frame, f"{label}: {len(saved_paths)}/{num_samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Collect", frame)
            last_frame_time = current_time
        
            if key == ord("q"):
                break
            if record:
                img_path = output_dir / f"{label}_{len(saved_paths):04d}.png"
                cv2.imwrite(str(img_path), frame)
                saved_paths.append(img_path)
                print(f"Saved {img_path}")

    camera.release()
    cv2.destroyAllWindows()
    return saved_paths


def extract_landmarks(image_paths: List[Path], label: str, window_size: int = 5) -> pd.DataFrame:
    """
    Extract landmarks using a sliding window of consecutive images.
    Each row contains features from window_size consecutive images.
    """
    tracker = HandTracker()
    rows = []
    
    # Process all images first to get their feature vectors
    all_features = []
    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        hands = tracker.process(image)
        if hands:
            features = landmarks_to_feature_vector(hands[0])
            all_features.append(features)
        else:
            # Store None for images without hands - we'll skip windows containing them
            all_features.append(None)
    
    # Create sliding window of window_size consecutive images
    for i in range(len(all_features) - window_size + 1):
        window_features = all_features[i:i + window_size]
        
        # Skip if any image in the window doesn't have hands
        if any(f is None for f in window_features):
            continue
        
        # Concatenate all feature vectors from the window
        concatenated_features = np.concatenate(window_features)
        
        # Create feature dictionary with all concatenated features
        feature_dict = {f"f{i}": val for i, val in enumerate(concatenated_features)}
        feature_dict["label"] = label
        rows.append(feature_dict)
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Collect gesture images and landmarks.")
    parser.add_argument("--label", required=True, help="Назва жесту / класу")
    parser.add_argument("--num", type=int, default=50, help="Кількість знімків")
    parser.add_argument("--raw_dir", type=Path, default=SRC_CANDIDATE.parent / "data/raw")
    parser.add_argument("--processed_csv", type=Path, default=SRC_CANDIDATE.parent / "data/processed/landmarks.csv")
    args = parser.parse_args()

    image_dir = args.raw_dir / args.label
    image_paths = collect_images(image_dir, args.label, args.num)
    if not image_paths:
        print("Не збережено жодного кадру.")
        return

    df = extract_landmarks(image_paths, args.label)
    if df.empty:
        print("Не вдалося витягнути точки MediaPipe.")
        return

    args.processed_csv.parent.mkdir(parents=True, exist_ok=True)

    # Якщо файл вже існує – просто додаємо нові рядки без перезапису заголовка.
    file_exists = args.processed_csv.exists()
    df.to_csv(
        args.processed_csv,
        mode="a" if file_exists else "w",
        header=not file_exists,
        index=False,
    )
    print(f"Додано {len(df)} нових записів у {args.processed_csv}")


if __name__ == "__main__":
    main()

