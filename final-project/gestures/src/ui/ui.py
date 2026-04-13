from __future__ import annotations

import PySimpleGUI as sg
import cv2
import numpy as np


class AppWindow:
    """Simple PySimpleGUI window to render frames and predictions."""

    def __init__(self, title: str = "SignSpeak"):
        sg.theme("DarkBlue3")
        layout = [
            [sg.Image(key="-IMAGE-")],
            [sg.Text("Розпізнаний жест:", size=(16, 1)), sg.Text("", key="-LABEL-")],
            [sg.Text("Ймовірність:", size=(16, 1)), sg.Text("", key="-CONF-")],
            [sg.Button("Вийти", key="-EXIT-")],
        ]
        self.window = sg.Window(title, layout, location=(200, 200))

    def update(self, frame_bgr: np.ndarray, label: str, confidence: float):
        imgbytes = cv2.imencode(".png", frame_bgr)[1].tobytes()
        self.window["-IMAGE-"].update(data=imgbytes)
        self.window["-LABEL-"].update(label)
        self.window["-CONF-"].update(f"{confidence:.2f}")

    def read(self, timeout: int = 1):
        return self.window.read(timeout=timeout)

    def close(self):
        self.window.close()

