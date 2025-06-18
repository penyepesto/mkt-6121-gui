from .classical_ops import get_operations as _g1
from .geometric_ops import get_operations as _g2
from .ai_ops        import get_operations as _g3

def all_operations():
    ops = {}
    for g in (_g1, _g2, _g3):
        ops.update(g())
    return ops
