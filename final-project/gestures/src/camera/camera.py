import cv2


class Camera:
    """Thin wrapper around OpenCV VideoCapture with basic safety checks."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int | None = None):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        return True

    @property
    def available(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        if not self.available:
            raise RuntimeError("Camera is not opened. Call open() first.")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera.")
        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

