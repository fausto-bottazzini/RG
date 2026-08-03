import sympy as sp
from Tensores.tensor import Tensor

class Christoffel(Tensor):
    """Calcula los simbolos de Christoffel de levi-civita"""

    def __init__(self, metrica):
        super().__init__(rank=3, dim=len(metrica.coords))
        self.metrica = metrica
        self.compute()

    def _gamma(self, dg, g_inv, mu, nu, lam):
        value = sp.Integer(0)
        for (idx, g_value) in g_inv.select({0: mu}):
            sigma = idx[1]
            value += (g_value * (dg[sigma, lam, nu] + dg[sigma, nu, lam] - dg[nu, lam, sigma]))
        return sp.Rational(1, 2) * value

    def compute(self):
        n = self.dim
        dg = self.metrica.derivadas()
        g_inv = self.metrica.g_inv
        for mu in range(n):
            for nu in range(n):
                for lam in range(nu, n):
                    gamma = self._gamma(dg, g_inv, mu, nu, lam)
                    if gamma != 0:
                        self[mu, nu, lam] = gamma

                        if nu != lam:
                            self[mu, lam, nu] = gamma
