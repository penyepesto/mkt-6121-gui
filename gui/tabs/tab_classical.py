# gui/tabs/tab_classical.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QMessageBox
from ..widgets.toggle_panel import TogglePanel
from ..widgets.param_panel  import ParamPanel
from ..processor import run_pipeline
from ..ops.classical_ops import get_operations as get_classical_ops


class ClassicalTab(QWidget):
    """‘Classical Methods’ sekmesi – renk uzayı, filtre, kenar, morfoloji …"""
    def __init__(self, input_manager, canvas, parent=None):
        super().__init__(parent)
        self.input  = input_manager   # InputManager referansı
        self.canvas = canvas          # Ortak ImageCanvas
        self.ops    = get_classical_ops()

        self.toggle = TogglePanel(self.ops)
        self.params = ParamPanel(self.ops)

        self.btn_run = QPushButton("Run Classical Ops")
        self.btn_run.clicked.connect(self._run)

        lay = QVBoxLayout(self)
        lay.addWidget(self.toggle)
        lay.addWidget(self.params)
        lay.addWidget(self.btn_run)
        lay.addStretch()

    # ------------------------------------------------------------------
    def _run(self):
        frame = self.input.next_frame()
        if frame is None:
            QMessageBox.warning(self, "No Input",
                                 "Önce bir resim / video yükleyin veya webcam açın.")
            return

        sel = self.toggle.selected_ops()
        param_map = {oid: self.params.params_for(oid) for oid in sel}

        out = run_pipeline(frame, sel, param_map, parent=self)

        self.canvas.clear()
        self.canvas.add_image(frame, "**Input**")
        self.canvas.add_image(out,   "**Output**")
