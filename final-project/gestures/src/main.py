from __future__ import annotations

import os
import re
import time
from functools import lru_cache
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import sys

from dotenv import load_dotenv

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore

try:
    import pyttsx3  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pyttsx3 = None

# Ensure the top-level src directory (which contains packages like camera, model, tracker, etc.)
# is on sys.path, regardless of where this script is executed from.
SRC_CANDIDATE = Path(__file__).resolve().parent
if not (SRC_CANDIDATE / "camera").is_dir():
    SRC_CANDIDATE = SRC_CANDIDATE.parent
if str(SRC_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(SRC_CANDIDATE))

from camera.camera import Camera
from train.gesture_model import GestureModel
from tracker.mediapipe_hand import HandTracker
from tracker.preprocessing import landmarks_to_feature_vector


def _safe_join(tokens: list[str]) -> str:
    return " ".join(t.strip() for t in tokens if t and t.strip())


@lru_cache(maxsize=8)
def _load_ui_font(size: int):
    if ImageFont is None:
        return None

    candidates = [
        os.getenv("UI_FONT_PATH", "").strip() or None,
        str(Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "arial.ttf"),
        str(Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "segoeui.ttf"),
    ]

    for path in candidates:
        if not path:
            continue
        try:
            if Path(path).exists():
                return ImageFont.truetype(path, size=size)
        except Exception:
            continue

    try:
        return ImageFont.load_default()
    except Exception:
        return None


def put_text_unicode(
    frame_bgr: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_size: int = 24,
    color_bgr: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """
    Draw Unicode text (e.g. Ukrainian) on a cv2 BGR frame.
    Uses Pillow if available; falls back to cv2.putText otherwise.
    """

    if not text.strip():
        return

    if Image is None or ImageDraw is None or ImageFont is None:
        cv2.putText(
            frame_bgr,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color_bgr,
            2,
        )
        return

    font = _load_ui_font(font_size)
    if font is None:
        return

    x, y = org
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    r, g, b = int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])
    # small shadow for readability
    draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(r, g, b))

    frame_bgr[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def generate_phrase_with_openai(tokens: list[str]) -> str:
    """
    Turn a short sequence of gesture labels into a short meaningful phrase.
    If OpenAI isn't configured or request fails, returns a plain concatenation.
    """

    sequence = _safe_join(tokens)
    if not sequence:
        return ""

    if len(tokens) <= 1:
        return sequence

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return sequence

    try:
        # Lazy import so the app can still run without OpenAI installed.
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"

        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=60,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ти перетворюєш короткий список жестових токенів на коротку змістовну фразу українською мовою. "
                        "Правила: використовуй ЛИШЕ надані токени, НЕ додавай нових слів, якщо в цьому немає потреби, але якщо без них фраза не має сенсу, то додавай слова за необхідності. Сформулюй коректне речення."
                        "НЕ розширюй фразу. Виводь ЛИШЕ готову фразу без лапок. Перестав слова за необхідності."
                        "Правила: 1) слово «how» («як») можна перекладати як «як» або «так» - залежно від контексту."
                        "2) слово «hello» («привіт») можна перекладати як «привіт» або «ні» - залежно від контексту."
                        "3) токени 's', 'a', 'sh' - літери 'с', 'а', 'ш' відповідно. Складай з них слово 'Саша'."
                        f"4) put_in -  це завжди у сенсі «покласти в»"
                    ),
                },
                {"role": "user", "content": f"Tokens: {sequence}"},
            ],
        )
        print(f"Request: {sequence}")
        print(f"Response: {resp.choices[0].message.content}")

        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return sequence


def maybe_init_tts():
    if pyttsx3 is None:
        return None
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        return engine
    except Exception:
        return None


def main():
    # Load OpenAI key from gestures/.env (not committed)
    load_dotenv(dotenv_path=SRC_CANDIDATE.parent / ".env")

    camera = Camera()
    if not camera.open():
        raise SystemExit("Не вдалося відкрити вебкамеру. Перевірте підключення.")

    tracker = HandTracker()
    model = GestureModel(
        model_path=SRC_CANDIDATE.parent / "models/gesture_model.joblib",
        labels_path=SRC_CANDIDATE.parent / "models/labels.json",
    )
    tts_engine = maybe_init_tts()
    last_label = None
    last_spoken_at = 0.0

    # Buffer consecutive gestures and generate a phrase after inactivity.
    phrase_idle_timeout_s = 5.0
    phrase_max_tokens = 10
    gesture_min_confidence = 0.3
    gesture_append_cooldown_s = 0.7  # avoid adding the same label every frame
    gesture_buffer: list[str] = []
    last_gesture_added_at = 0.0
    last_buffer_activity_at = time.time()

    latest_phrase = ""
    phrase_status = ""  # "", "…"
    executor = ThreadPoolExecutor(max_workers=1)
    pending_future: Future[str] | None = None
    queued_batches: deque[list[str]] = deque()

    def submit_phrase_job(tokens: list[str]) -> None:
        nonlocal pending_future, phrase_status
        if not tokens:
            return
        if pending_future is None:
            pending_future = executor.submit(generate_phrase_with_openai, tokens)
            phrase_status = "…"
        else:
            queued_batches.append(tokens)

    # Frame rate control (5 FPS)
    target_fps = 5
    frame_time = 1.0 / target_fps  # 0.2 seconds per frame
    last_frame_time = time.time()
    
    # Sliding window buffer for 5 consecutive frames
    window_size = 5
    feature_buffer = []

    print("Натисніть 'q' для виходу.")
    while True:
        current_time = time.time()
        elapsed = current_time - last_frame_time

        # Always check for keyboard input to keep UI responsive
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # Only process and display frame if enough time has passed (5 FPS)
        if elapsed >= frame_time:
            frame = camera.read()
            hands = tracker.process(frame)
            label, confidence = "<no signal>", 0.0

            # Collect finished OpenAI results without blocking the UI.
            if pending_future is not None and pending_future.done():
                try:
                    latest_phrase = pending_future.result()
                    print(f"Latest phrase: {latest_phrase}")
                except Exception:
                    pass
                pending_future = None
                phrase_status = ""

                if queued_batches:
                    pending_future = executor.submit(generate_phrase_with_openai, queued_batches.popleft())
                    phrase_status = "…"

            if hands:
                tracker.draw(frame, hands)
                features = landmarks_to_feature_vector(hands[0])
                
                # Add current frame features to buffer
                feature_buffer.append(features)
                
                # Keep only the last window_size frames
                if len(feature_buffer) > window_size:
                    feature_buffer.pop(0)
                
                # Make prediction only when we have window_size frames
                if len(feature_buffer) == window_size:
                    # Concatenate features from all 5 frames (same as extract_landmarks)
                    concatenated_features = np.concatenate(feature_buffer)
                    label, confidence = model.predict(concatenated_features)

                    if tts_engine and label != last_label and confidence > 0.6:
                        now = time.time()
                        if now - last_spoken_at > 1.5:
                            tts_engine.say(label)
                            tts_engine.runAndWait()
                            last_label = label
                            last_spoken_at = now

                    # Append new gesture label into sequence buffer (deduped + debounced).
                    if confidence >= gesture_min_confidence and label not in ("<no signal>", "unknown"):
                        now = time.time()
                        print(f"Gesture buffer: {gesture_buffer}, label: {label}")
                        can_append = (
                            ((not gesture_buffer) or (label != gesture_buffer[-1]))
                            and (now - last_gesture_added_at >= gesture_append_cooldown_s)
                        )
                        if can_append:
                            gesture_buffer.append(label)
                            last_gesture_added_at = now
                            last_buffer_activity_at = now

                            # Flush immediately if buffer is full.
                            if len(gesture_buffer) >= phrase_max_tokens:
                                tokens = gesture_buffer[:]
                                gesture_buffer.clear()
                                last_gesture_added_at = 0.0
                                submit_phrase_job(tokens)
            else:
                # Reset buffer if no hands detected
                feature_buffer.clear()

            # If we've had no NEW gesture tokens for a while, generate the phrase.
            now = time.time()
            if gesture_buffer and (now - last_buffer_activity_at >= phrase_idle_timeout_s):
                tokens = gesture_buffer[:]
                gesture_buffer.clear()
                last_gesture_added_at = 0.0
                submit_phrase_job(tokens)

            put_text_unicode(frame, f"{label} ({confidence:.2f})", (10, 10), font_size=28, color_bgr=(0, 255, 0))
            put_text_unicode(
                frame,
                f"Tokens: {len(gesture_buffer)}/{phrase_max_tokens}",
                (10, 45),
                font_size=22,
                color_bgr=(255, 255, 0),
            )

            phrase_text = latest_phrase + (f" {phrase_status}" if phrase_status else "")
            if phrase_text.strip():
                put_text_unicode(frame, phrase_text.strip(), (10, 80), font_size=26, color_bgr=(255, 255, 255))

            cv2.imshow("SignSpeak", frame)
            last_frame_time = current_time

    camera.release()
    cv2.destroyAllWindows()
    executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()

