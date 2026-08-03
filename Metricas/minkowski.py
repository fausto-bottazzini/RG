import sympy as sp
from .metrica import Metrica
from Tensores.tensor import Tensor

class Minkowski(Metrica):
    def __init__(self):
        super().__init__()
        t, x, y, z = sp.symbols("t x y z", real=True)
        self.coords = (t, x, y, z)

    def tensor_metrico(self):
        return Tensor.from_matrix(
            sp.diag(-1, 1, 1, 1))