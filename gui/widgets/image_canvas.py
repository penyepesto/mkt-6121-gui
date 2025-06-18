from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget, QScrollArea
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import cv2

class ImageCanvas(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        w = QWidget()
        self.vb = QVBoxLayout(w)
        self.setWidget(w); self.setWidgetResizable(True)

    def clear(self):
        while self.vb.count():
            child = self.vb.takeAt(0).widget()
            if child:
                child.setParent(None)

    def add_image(self, img, title=""):
        h, w, _ = img.shape
        q = QImage(img.data, w, h, 3 * w, QImage.Format_BGR888)
        pix = QPixmap.fromImage(q)
        lbl = QLabel(title)
        pic = QLabel(); pic.setPixmap(pix)
        self.vb.addWidget(lbl)
        self.vb.addWidget(pic)
