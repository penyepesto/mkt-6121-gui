# gui/widgets/param_panel.py
from PyQt5.QtWidgets import (QWidget, QFormLayout, QSpinBox, QDoubleSpinBox,
                             QComboBox, QColorDialog, QPushButton)
from ..ops import all_operations
from PyQt5.QtWidgets import QSpacerItem, QSizePolicy


class ParamPanel(QWidget):
    def __init__(self, ops=None, parent=None):
        super().__init__(parent)
        self.ops = ops or all_operations()
        self.controls = {}        # (op_id, param) → widget

        self.form = QFormLayout(self)
        for sp in self.ops.values():
            for pname, meta in sp.controls.items():
                w = self._make_widget(meta)
                self.form.addRow(f"{sp.display} · {pname}", w)
                self.controls[(sp.id, pname)] = w

        #self.form.addStretch()
        self.form.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

    def _make_widget(self, m):
        t = m["type"]
        if t == "spin":
            w = QSpinBox(); w.setRange(m["min"], m["max"]); w.setSingleStep(m.get("step",1)); w.setValue(m["default"])
        elif t == "dspin":
            w = QDoubleSpinBox(); w.setDecimals(2)
            w.setRange(m["min"], m["max"]); w.setSingleStep(m.get("step",0.1)); w.setValue(m["default"])
        elif t == "slider":
            from PyQt5.QtWidgets import QSlider
            w = QSlider(1); w.setRange(m["min"], m["max"]); w.setSingleStep(m.get("step",1)); w.setValue(m["default"])
        elif t == "combo":
            w = QComboBox(); w.addItems(m["items"]); w.setCurrentText(m["default"])
        elif t == "color":
            w = QPushButton("Pick"); w.clr = m["default"]
            w.clicked.connect(lambda _, b=w: self._pick_color(b))
        elif t == "bool":
            from PyQt5.QtWidgets import QCheckBox
            w = QCheckBox(); w.setChecked(m["default"])
        elif t == "file":
            from PyQt5.QtWidgets import QPushButton
            w = QPushButton(m["default"])
            w.clicked.connect(lambda _,b=w: self._select_file(b))

        else:
            raise ValueError(t)
        return w

    def _pick_color(self, btn):
        c = QColorDialog.getColor()
        if c.isValid():
            btn.clr = (c.red(), c.green(), c.blue())
            btn.setStyleSheet(f"background-color: {c.name()};")

    def params_for(self, op_id):
        sp = self.ops[op_id]
        d = {}
        for pname in sp.controls:
            w = self.controls[(op_id, pname)]
            if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                d[pname] = w.value()
            elif isinstance(w, QComboBox):
                d[pname] = w.currentText()
            elif hasattr(w, "clr"):
                d[pname] = w.clr
            else:   # slider
                d[pname] = w.value()
        return d
