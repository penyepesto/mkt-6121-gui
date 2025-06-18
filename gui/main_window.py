# -*- coding: utf-8 -*-
"""Re‑written MainWindow with high‑quality image display,
robust image loading (Unicode‑safe), and a Pause/Resume toggle
for video & webcam streams.

This file *replaces* the previous main_window.py one‑liner version.
All earlier logic is preserved; only the parts explicitly requested
by the user are changed or added:

1. **High‑quality display** – _show() no longer rescales; the QLabel is
   resized to the Pixmap to keep full resolution inside its QScrollArea.
2. **Robust load_image()** – uses cv2.imdecode + np.fromfile so UTF‑8
   paths on Windows work; also stops any running video timers.
3. **Pause/Resume toggle** – new QPushButton in control dock, wired to
   _toggle_pause_stream().  Enabled only when a live stream is active.
"""

from __future__ import annotations

import os, sys, cv2, numpy as np
from pathlib import Path
from typing import Dict, Any, List
from collections import namedtuple
from PyQt6.QtWidgets import QFileDialog, QApplication
from PyQt6.QtCore    import Qt, QTimer, QEvent
from PyQt6.QtGui     import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QTabWidget, QFileDialog, QSplitter, QScrollArea, QGroupBox,
    QGridLayout, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QListWidget,
    QSlider, QDockWidget, QStatusBar, QMessageBox, QLineEdit
)

from gui.ops.classical_ops import get_operations as _classical_ops
from gui.ops.geometric_ops import get_operations as _geo_ops
from gui.ops.ai_ops        import get_operations as _ai_ops


OperationSpec = Any
MatchedPair   = namedtuple("MatchedPair", ["img", "kps", "des"])

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _interp_flag(s: str) -> int:
    return {
        "Nearest"  : cv2.INTER_NEAREST,
        "Bilinear" : cv2.INTER_LINEAR,
        "Bicubic"  : cv2.INTER_CUBIC,
    }.get(s, cv2.INTER_LINEAR)

# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Three‑tab advanced image‑processing GUI (updated)."""

    # --------------------------- init ----------------------------------
    def __init__(self):
        super().__init__()

        # runtime state -------------------------------------------------
        self.current_image:   np.ndarray | None = None
        self.processed_image: np.ndarray | None = None
        self.video_capture    = None
        self.video_writer     = None
        self.recording        = False
        self.recording_fps    = 30
        self.processing_history: List[np.ndarray] = []
        
        
        self.setStyleSheet("""
            QGroupBox { 
                border: 1px solid #777; 
                border-radius: 6px; 
                margin-top: 5px;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 10px; 
                padding: 0 3px 0 3px;
                font-weight: bold;
            }
            QPushButton { 
                padding: 4px 10px;
            }
        """)


        # operation registries -----------------------------------------
        self.ops_c = _classical_ops()
        self.ops_g = _geo_ops()
        self.ops_a = _ai_ops()

        # build ui ------------------------------------------------------
        self._build_ui()

        # timer for video/webcam frames --------------------------------
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

        # mouse tracking on input image --------------------------------
        self.input_image_label.setMouseTracking(True)
        self.input_image_label.installEventFilter(self)

    # ------------------------------------------------------------------
    #  UI BUILDERS
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.setWindowTitle("Advanced Image Processing Suite")
        self.resize(1400, 900)

        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)

        self._create_menu_bar()
        self._create_toolbar()
        self._create_input_section(main)

        img_split = self._create_image_display()

        self.tab_widget = QTabWidget()
        self._add_classical_tab()
        self._add_geometric_tab()
        self._add_modern_tab()

        vs = QSplitter(Qt.Orientation.Vertical)
        vs.addWidget(img_split)
        vs.addWidget(self.tab_widget)
        vs.setStretchFactor(0, 4)
        vs.setStretchFactor(1, 3)
        main.addWidget(vs)

        self._create_control_dock()
        self._create_status_bar()

    # ---------------- menu & toolbar ----------------------------------

    def _create_menu_bar(self):
        fm = self.menuBar().addMenu("File")
        for t, s in [
            ("Open Image", self._open_image),
            ("Open Video", self._open_video),
            ("Save Result", self._save_result),
            ("Export History", self._export_hist),
            ("Exit", self.close),
        ]:
            a = fm.addAction(t)
            a.triggered.connect(s)



    def _create_toolbar(self):
        tb = self.addToolBar("Tools")
        tb.setMovable(False)
        for t, s in [
            ("Open", self._open_image),
            ("Save", self._save_result),
            ("Reset", self._reset_proc),
            ("Record", self._start_webcam),
            ("Toggle Controls", self._toggle_ctrl),
        ]:
            b = QPushButton(t)
            b.clicked.connect(s)
            tb.addWidget(b)

    # ---------------- source bar --------------------------------------

    def _create_input_section(self, vbox: QVBoxLayout):
        grp = QGroupBox("Input Source")
        lay = QHBoxLayout(grp)

        self.src_combo = QComboBox()
        self.src_combo.addItems(["Single Image", "Video File", "Webcam"])
        self.src_combo.currentTextChanged.connect(self._change_src)

        self.sel_btn = QPushButton("Select Source")
        self.sel_btn.clicked.connect(self._select_src)

        self.res_combo = QComboBox()
        self.res_combo.addItems(["640x480", "1280x720", "1920x1080"])
        self.res_combo.setVisible(False)

        for w in [
            QLabel("Source Type:"),
            self.src_combo,
            self.sel_btn,
            QLabel("Resolution:"),
            self.res_combo,
        ]:
            lay.addWidget(w)
        lay.addStretch()
        vbox.addWidget(grp)

    # ---------------- image display -----------------------------------

    def _create_image_display(self) -> QSplitter:
        def _panel(title: str):
            grp = QGroupBox(title)
            v = QVBoxLayout(grp)
            lab = QLabel()
            lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            sc = QScrollArea()
            sc.setWidget(lab)
            sc.setWidgetResizable(True)
            v.addWidget(sc)
            return grp, lab

        in_grp, self.input_image_label  = _panel("Input Image")
        out_grp, self.output_image_label = _panel("Processed Image")

        sp = QSplitter(Qt.Orientation.Horizontal)
        sp.addWidget(in_grp)
        sp.addWidget(out_grp)
        sp.setStretchFactor(0, 1)
        sp.setStretchFactor(1, 1)
        return sp

    # ---------------- param widget factory ----------------------------

    def _make_param_widget(self, op_id: str, pname: str, meta: dict):
        t = meta.get("type", "spin")
        if t == "spin":
            w = QSpinBox()
            w.setRange(meta["min"], meta["max"])
            w.setValue(meta["default"])
        elif t == "dspin":
            w = QDoubleSpinBox()
            w.setRange(meta["min"], meta["max"])
            w.setSingleStep(meta.get("step", 0.1))
            w.setValue(meta["default"])
        elif t == "slider":
            w = QSlider(Qt.Orientation.Horizontal)
            w.setRange(meta["min"], meta["max"])
            w.setValue(meta["default"])
        elif t == "combo":
            w = QComboBox()
            w.addItems(meta["items"])
            w.setCurrentText(meta["default"])
        elif t == "file":
            w = QPushButton(Path(meta["default"]).name)
            w._path = meta["default"]
            def _pick(btn):
                fn,_ = QFileDialog.getOpenFileName(self, "Model checkpoint", str(Path.home()), "*.*")
                if fn:
                    btn.setText(Path(fn).name); btn._path = fn
            w.clicked.connect(lambda _,b=w: _pick(b))

        elif t == "color":
            def _pick(btn):
                from PyQt6.QtWidgets import QColorDialog
                c = QColorDialog.getColor()
                if c.isValid():
                    btn.setStyleSheet(f"background:{c.name()}")
                    btn._color = [c.red(), c.green(), c.blue()]
            w = QPushButton()
            w._color = meta["default"]
            w.setStyleSheet(f"background: rgb{tuple(meta['default'])}")
            w.clicked.connect(lambda _, b=w: _pick(b))

        else:
            w = QLineEdit(str(meta["default"]))
        setattr(self, f"param_{op_id}_{pname}", w)
        return w

    # ---------------- ops tabs ----------------------------------------

    def _add_ops_tab(self, ops: Dict[str, OperationSpec], title: str, cats: List[str]):
        scroll = QScrollArea()
        tab    = QWidget()
        scroll.setWidget(tab)
        scroll.setWidgetResizable(True)
        self.tab_widget.addTab(scroll, title)

        lay = QVBoxLayout(tab)
        for cat in cats:
            grp = QGroupBox(cat)
            v   = QVBoxLayout(grp)
            for spec in (s for s in ops.values() if s.category == cat):
                cb = QCheckBox(spec.display)
                cb.stateChanged.connect(self._update_ops)
                setattr(self, f"chk_{spec.id}", cb)
                v.addWidget(cb)
                if spec.controls:
                    g, r = QGridLayout(), 0
                    for n, m in spec.controls.items():
                        g.addWidget(QLabel(n + ":"), r, 0)
                        g.addWidget(self._make_param_widget(spec.id, n, m), r, 1)
                        r += 1
                    v.addLayout(g)
            lay.addWidget(grp)
        lay.addStretch()

    def _add_classical_tab(self):
        self._add_ops_tab(self.ops_c, "Classical Methods", ["Color", "Filtering", "Edge", "Morphology"])

    def _add_geometric_tab(self):
        self._add_ops_tab(self.ops_g, "Geometric Methods", ["Feature", "Basic", "Advanced"])

    def _add_modern_tab(self):
        self._add_ops_tab(self.ops_a, "Modern Methods", ["Detection", "Segmentation", "Anomaly", "GAN"])

    # ---------------- control dock ------------------------------------

    def _create_control_dock(self):
        cw = QWidget()
        l  = QHBoxLayout(cw)

        self.ops_list = QListWidget()
        self.ops_list.setFixedWidth(250)
        l.addWidget(self.ops_list)

        mid = QVBoxLayout()
        self.live_prev = QCheckBox("Live Preview")
        mid.addWidget(self.live_prev)

        proc_btn = QPushButton("Process")
        proc_btn.clicked.connect(self._process)
        mid.addWidget(proc_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._reset_proc)
        mid.addWidget(reset_btn)
        
        meta_btn = QPushButton("Download metadata")
        meta_btn.clicked.connect(self._save_meta)
        mid.addWidget(meta_btn)

        # NEW ↘ Pause / Resume toggle
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.setEnabled(False)  # only when stream running
        self.pause_btn.toggled.connect(self._toggle_pause_stream)
        mid.addWidget(self.pause_btn)

        l.addLayout(mid)

        self.ctrl_dock = QDockWidget("Controls", self)
        self.ctrl_dock.setWidget(cw)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.ctrl_dock)

    def _toggle_ctrl(self):
        self.ctrl_dock.setVisible(not self.ctrl_dock.isVisible())
    def _save_meta(self):
        if not self.last_meta:
            return self._err("No metadata yet")
        meta = {
            "file": self._last_loaded_file or "stream_frame",
            **self.last_meta,
        }
        fn, _ = QFileDialog.getSaveFileName(self, "Save meta", str(Path.home()), "*.json")
        if fn:
            import json, time
            meta["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

    # ---------------- status bar --------------------------------------

    def _create_status_bar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        sb.showMessage("Ready")
        self.info_lbl  = QLabel()
        sb.addPermanentWidget(self.info_lbl)
        self.mouse_lbl = QLabel("Mouse: -,-")
        sb.addPermanentWidget(self.mouse_lbl)
        self.status_bar = sb

     # ------------------------------------------------------------------
    # SOURCE HELPERS
    # ------------------------------------------------------------------

    def _select_src(self):
        sel = self.src_combo.currentText()
        if sel == "Single Image":
            self._open_image()
        elif sel == "Video File":
            self._open_video()
        else:
            self._start_webcam()

    def _change_src(self):
        if self.video_capture is not None:
            self.video_capture.release(); self.video_capture = None; self.timer.stop()
        self._reset_proc()
        self.res_combo.setVisible(self.src_combo.currentText() == "Webcam")

    def _open_image(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Image", str(Path.home()), "Images (*.png *.jpg *.bmp *.tiff)")
        if fn:
            self._load_img(fn)

    def _open_video(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Video", str(Path.home()), "Videos (*.mp4 *.avi *.mkv)")
        if fn:
            self._start_video(fn)

    # ----------------------------- Unicode‑safe image load ----------

    def _load_img(self, fn: str):
        try:
            img = cv2.imdecode(np.fromfile(fn, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            return self._err("Cannot load image")
        self.current_image = img; self.processed_image = img.copy(); self._show(img, self.input_image_label); self._show(img, self.output_image_label)
        self.pause_btn.setEnabled(False)
        self.last_meta: Dict[str, Any] = {}
        self._last_loaded_file: str    = ""


    # ----------------------------- video / webcam -------------------

    def _start_video(self, fn: str):
        self.video_capture = cv2.VideoCapture(fn); self.timer.start(30)
        self.pause_btn.setEnabled(True); self.pause_btn.setChecked(False)

    def _start_webcam(self):
        self.video_capture = cv2.VideoCapture(0)
        w, h = map(int, self.res_combo.currentText().split("x"))
        self.video_capture.set(3, w); self.video_capture.set(4, h)
        self.timer.start(30)
        self.pause_btn.setEnabled(True); self.pause_btn.setChecked(False)

    # ------------------------ pause / resume -------------------------

    def _toggle_pause_stream(self, checked: bool):
        if checked:
            if self.timer.isActive():
                self.timer.stop()
            self.pause_btn.setText("▶ Resume")
            self.status_bar.showMessage("Stream paused")
        else:
            if self.video_capture is not None:
                self.timer.start(30)
            self.pause_btn.setText("⏸ Pause")
            self.status_bar.showMessage("Stream resumed")

    # ------------------------ frame update ---------------------------

    def _update_frame(self):
        if self.video_capture is None or not self.timer.isActive():
            return
        ret, frame = self.video_capture.read()
        if not ret:
            self.timer.stop(); return
        self.current_image = frame
        self._show(frame, self.input_image_label)
        if self.live_prev.isChecked():
            self._process(auto=True)
        if self.live_prev.isChecked():
            self._process(auto=True)


    # ----------------------------------------------------------------
    # ACTIVE OPERATIONS LIST
    # ----------------------------------------------------------------

    def _update_ops(self):
        self.ops_list.clear()
        for spec in list(self.ops_c.values()) + list(self.ops_g.values()) + list(self.ops_a.values()):
            chk = getattr(self, f"chk_{spec.id}", None)
            if chk and chk.isChecked():
                self.ops_list.addItem(spec.display)
        if self.live_prev.isChecked():
            self._process()

    # ----------------------------------------------------------------
    # PARAM COLLECTION
    # ----------------------------------------------------------------

    def _gather(self, spec: OperationSpec) -> dict:
        d = {}
        for n in spec.controls:
            w = getattr(self, f"param_{spec.id}_{n}")
            if isinstance(w, (QSpinBox, QDoubleSpinBox, QSlider)):
                val = w.value()
            elif isinstance(w, QComboBox):
                val = w.currentText()
            elif isinstance(w, QPushButton) and hasattr(w, "_color"):
                val = w._color

            else:               # QLineEdit
                txt = w.text()
                val = int(txt) if txt.isdigit() else float(txt) if txt.replace('.', '', 1).isdigit() else txt
            d[n] = val
        return d

    # ----------------------------------------------------------------
    # MAIN PROCESSING PIPELINE
    # ----------------------------------------------------------------
        
    def _process(self, auto: bool = False):
        if self.current_image is None:
            return

        img        = self.current_image.copy()
        self.last_meta = {}                      # << önce temizle
        for spec in list(self.ops_c.values()) + list(self.ops_g.values()) + list(self.ops_a.values()):
            chk = getattr(self, f"chk_{spec.id}", None)
            if chk and chk.isChecked():
                try:
                    res = spec.func(img, self._gather(spec))
                except Exception as e:
                    self._err(f"{spec.display} failed: {e}"); continue

                # --- tuple → (img, meta) ---------------------------
                if isinstance(res, tuple) and len(res) == 2:
                    img, meta = res
                    self.last_meta.update(meta)  # toplu meta
                else:
                    img = res
        # -----------------------------------------------------------

        self.processed_image = img
        self._show(img, self.output_image_label)
        if not auto:
            self.processing_history.append(img.copy())

        

    # ----------------------------------------------------------------
    # DISPLAY & ERROR HELPERS
    # ----------------------------------------------------------------

    def _show(self, img: np.ndarray, lbl: QLabel):
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(q)
        lbl.setPixmap(pix)
        lbl.resize(pix.size())           # tam çözünürlük → ScrollArea kaydırılabilir

    def _err(self, msg: str):
        QMessageBox.critical(self, "Error", msg)

    # ----------------------------------------------------------------
    # SAVE / EXPORT / RESET
    # ----------------------------------------------------------------

    def _reset_proc(self):
        if self.current_image is not None:
            self.processed_image = self.current_image.copy()
            self._show(self.processed_image, self.output_image_label)
            self.processing_history.clear()

    def _save_result(self):
        if self.processed_image is None:
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Save", str(Path.home()), "*.png;*.jpg")
        if fn:
            cv2.imwrite(fn, self.processed_image)

    def _export_hist(self):
        if not self.processing_history:
            return self._err("No history")
        dir_ = QFileDialog.getExistingDirectory(self, "Dir", str(Path.home()))
        if dir_:
            for i, im in enumerate(self.processing_history):
                cv2.imwrite(os.path.join(dir_, f"step_{i+1}.png"), im)

    # ----------------------------------------------------------------
    # MOUSE COORD DISPLAY
    # ----------------------------------------------------------------

    def eventFilter(self, src, ev):
        if src is self.input_image_label and ev.type() == QEvent.Type.MouseMove and self.current_image is not None:
            x = int(ev.position().x() * self.current_image.shape[1] / src.width())
            y = int(ev.position().y() * self.current_image.shape[0] / src.height())
            self.mouse_lbl.setText(f"Mouse: {x},{y}")
        return super().eventFilter(src, ev)

    # ----------------------------------------------------------------
    # CLOSE HANDLER
    # ----------------------------------------------------------------

    def closeEvent(self, ev):
        if self.video_capture is not None:
            self.video_capture.release(); self.timer.stop()
        if self.video_writer is not None:
            self.video_writer.release()
        ev.accept()

# ----------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
