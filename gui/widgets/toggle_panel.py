# gui/widgets/toggle_panel.py
from PyQt5.QtWidgets import (QGroupBox, QScrollArea, QWidget, QVBoxLayout,
                             QCheckBox)
from collections import defaultdict
from ..ops import all_operations

class TogglePanel(QWidget):
    def __init__(self, ops=None, parent=None):
        super().__init__(parent)
        self.ops = ops or all_operations()
        self.checks = {}             # op_id → QCheckBox
        layout = QVBoxLayout(self)

        # 1) Kategori bazlı grupla
        cat_map = defaultdict(list)
        for spec in self.ops.values():
            cat_map[spec.category].append(spec)

        for cat, specs in cat_map.items():
            gb = QGroupBox(cat)
            vb = QVBoxLayout(gb)
            for sp in sorted(specs, key=lambda s: s.display):
                cb = QCheckBox(sp.display)
                self.checks[sp.id] = cb
                vb.addWidget(cb)
            gb.setLayout(vb)
            layout.addWidget(gb)
        layout.addStretch()

    def selected_ops(self):
        return [oid for oid, cb in self.checks.items() if cb.isChecked()]
