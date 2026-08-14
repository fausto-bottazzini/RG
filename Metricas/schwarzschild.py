import sympy as sp
from Tensores.metrica import Metrica
from Tensores.tensor import Tensor

class Schwarzschild(Metrica):
    def __init__(self, M):
        super().__init__()
        t, r, theta, phi = sp.symbols('t r theta phi', real = True)
        self.coords = (t, r, theta, phi)
        self.params["M"] = M 

    def tensor_metrico(self):
        M = self.params["M"]
        _, r, theta, _ = self.coords
        f = 1 - (2 * M) / r
        return Tensor.from_matrix(sp.diag(-f, 1/f, r**2, r**2*sp.sin(theta)**2))