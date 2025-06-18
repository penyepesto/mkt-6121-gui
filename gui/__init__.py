from .main_window import MainWindow

__all__ = ["MainWindow"]


from .ops.classical_ops import get_operations as _c
from .ops.geometric_ops import get_operations as _g
from .ops.ai_ops        import get_operations as _a

def all_operations():
    d = {}
    for m in (_c(), _g(), _a()):
        d.update(m)
    return d
