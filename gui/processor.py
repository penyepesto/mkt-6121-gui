# gui/processor.py
import traceback
from PyQt5.QtWidgets import QMessageBox
from .ops import all_operations

OPS = all_operations()

def run_pipeline(img, selected, params, parent=None):
    """
    img: numpy.ndarray kaynak frame
    selected: [op_id, op_id, …]  # Toggle panel sırası
    params:   {op_id: {param:val}}
    """
    out = img
    for oid in selected:
        try:
            out = OPS[oid].func(out, params.get(oid, {}))
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(parent, "Processing Error",
                                 f"{OPS[oid].display} failed:\n{e}")
            return out    
    return out
