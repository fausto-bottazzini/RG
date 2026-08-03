import sympy as sp
from .metrica import Metrica
from Tensores.tensor import Tensor

class FLRW(Metrica):
    def __init__(self, k=0):
        super().__init__()
        t, r, theta, phi = sp.symbols("t r theta phi", real=True)
        a = sp.Function("a")(t)
        self.coords = (t, r, theta, phi)
        self.params["k"] = k
        self.params["a"] = a

    def tensor_metrico(self):
        k = self.params["k"]
        a = self.params["a"]
        _, r, theta, _ = self.coords
        return Tensor.from_matrix(sp.diag(-1, a**2/(1-k*r**2), a**2*r**2, a**2*r**2*sp.sin(theta)**2))