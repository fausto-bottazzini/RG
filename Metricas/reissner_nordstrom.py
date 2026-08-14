import sympy as sp
from Tensores.metrica import Metrica
from Tensores.tensor import Tensor

class ReissnerNordstrom(Metrica):
    def __init__(self, M, Q):
        super().__init__()
        t, r, theta, phi = sp.symbols("t r theta phi", real=True)
        self.coords = (t, r, theta, phi)
        self.params["M"] = M
        self.params["Q"] = Q

    def tensor_metrico(self):
        M = self.params["M"]
        Q = self.params["Q"]
        _, r, theta, _ = self.coords
        f = 1 - 2*M/r + Q**2/r**2
        return Tensor.from_matrix(
            sp.diag(-f, 1/f, r**2, r**2*sp.sin(theta)**2)
        )