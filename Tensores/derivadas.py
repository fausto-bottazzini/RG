import sympy as sp
from Tensores.tensor import Tensor

class DerivadasMetrica(Tensor):
    """Calculo de las derivadas parciales no nulas del tensor metrico"""

    def __init__(self, metrica):
        super().__init__(rank=3, dim=len(metrica.coords))
        self.metrica = metrica
        self.compute()

    def compute(self):
        coords = self.metrica.coords

        for (i,j), gij in self.metrica.g.items():

            variables = frozenset(gij.free_symbols)
            for k, coord in enumerate(coords):
                if coord not in variables:
                    continue

                dg = sp.diff(gij, coord)

                if dg == 0:
                    continue

                self[i,j,k] = dg
                if i != j:
                    self[j,i,k] = dg


class DerivadasChristoffel(Tensor):
    """Calculo de las derivadas parciales no nulas de los simbolos de Christoffel"""

    def __init__(self, metrica):
        super().__init__(rank=4, dim=len(metrica.coords))
        self.metrica = metrica
        self.compute()

    def compute(self):
        coords = self.metrica.coords
        Gamma = self.metrica.christoffel()

        for (mu, nu, lam), gamma in Gamma.items():
            variables = frozenset(gamma.free_symbols)
            for alpha, coord in enumerate(coords):
                if coord not in variables:
                    continue

                dgamma = sp.diff(gamma, coord)

                if dgamma == 0:
                    continue

                self[mu, nu, lam, alpha] = dgamma