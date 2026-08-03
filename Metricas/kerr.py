import sympy as sp
from .metrica import Metrica
from Tensores.tensor import Tensor

class Kerr(Metrica):
    def __init__(self, M, a):
        super().__init__()
        t, r, theta, phi = sp.symbols("t r theta phi", real=True)
        self.coords = (t, r, theta, phi)
        self.params["M"] = M
        self.params["a"] = a

    def tensor_metrico(self):
        M = self.params["M"]
        a = self.params["a"]
        _, r, theta, _ = self.coords

        Sigma = r**2 + a**2*sp.cos(theta)**2
        Delta = r**2 - 2*M*r + a**2
        g_tt = -(1 - 2*M*r/Sigma)
        g_rr = Sigma/Delta
        g_thth = Sigma
        g_phph = ((r**2 + a**2 + 2*M*r*a**2*sp.sin(theta)**2/Sigma)* sp.sin(theta)**2)
        g_tph = -2*M*r*a*sp.sin(theta)**2/Sigma

        return Tensor.from_matrix(
            sp.Matrix([
                [g_tt,   0,      0,      g_tph],
                [0,      g_rr,   0,      0],
                [0,      0,      g_thth, 0],
                [g_tph,  0,      0,      g_phph]
            ])
        )