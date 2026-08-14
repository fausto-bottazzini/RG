import sympy as sp
from Tensores.metrica import Metrica
from Tensores.tensor import Tensor

class AntiDeSitter(Metrica):
    def __init__(self, Lambda):
        super().__init__()
        t, r, theta, phi = sp.symbols("t r theta phi", real=True)
        self.coords = (t, r, theta, phi)
        self.params["Lambda"] = Lambda

    def tensor_metrico(self):
        Lambda = self.params["Lambda"]
        _, r, theta, _ = self.coords
        f = 1 + Lambda*r**2/3
        return Tensor.from_matrix(sp.diag(-f, 1/f, r**2, r**2*sp.sin(theta)**2))