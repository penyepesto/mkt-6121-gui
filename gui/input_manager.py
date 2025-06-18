import cv2

class InputManager:
    def __init__(self):
        self.cap = None        
        self.kind = "image"     # "image" | "video" | "webcam"
        self.frame = None

    def load_image(self, path):
        self.frame = cv2.imread(path)
        self.kind = "image"
        if self.frame is None:
            raise IOError("Cannot load image")

    def load_video(self, path):
        self.cap = cv2.VideoCapture(path)
        self.kind = "video"
        if not self.cap.isOpened():
            raise IOError("Cannot open video")

    def open_webcam(self, idx=0):
        self.cap = cv2.VideoCapture(idx)
        self.kind = "webcam"
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

    def next_frame(self):
        if self.kind == "image":
            return self.frame
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap:
            self.cap.release()
