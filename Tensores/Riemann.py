import sympy as sp
from Tensores.tensor import Tensor

class Riemann(Tensor):
    """Clase para el tensor de Riemann"""
    def __init__(self, metrica):
        super().__init__(rank=4, dim=len(metrica.coords))
        self.metrica = metrica
        self.compute()

    def _riemann(self, rho, sigma, mu, nu):
        value = sp.Integer(0)
        Gamma = self.metrica.christoffel()
        dGamma = self.metrica.derivadas_christoffel()

        value += dGamma[rho, nu, sigma, mu] 
        value -= dGamma[rho, mu, sigma, nu]

        for idx, gamma in Gamma.select({0: rho, 1:mu}):
            lam = idx[2]
            value += gamma * Gamma[lam,nu,sigma]


        for idx, gamma in Gamma.select({0: rho, 1: nu}):
            lam = idx[2]
            value -= gamma * Gamma[lam, mu, sigma]

        return value

    def compute(self):
        candidates = set()

        Gamma = self.metrica.christoffel()
        dGamma = self.metrica.derivadas_christoffel()

        # ---- términos con derivadas ----
        for (rho, sigma, mu, alpha), value in dGamma.items():
            candidates.add((rho, sigma, alpha, mu))
            candidates.add((rho, sigma, mu, alpha))

        # ---- términos Gamma Gamma ----
        for idx1, gamma1 in Gamma.items():         
            rho, mu, lam = idx1
            for idx2, gamma2 in Gamma.select({0: lam}):
                _, nu, sigma = idx2
                candidates.add((rho, sigma, mu, nu))
                candidates.add((rho, sigma, nu, mu))

        # ---- calcular solamente candidatos ----
        for rho, sigma, mu, nu in candidates:
            if mu > nu:
                continue

            R = self._riemann(rho, sigma, mu, nu)

            if R != 0:
                self[rho,sigma,mu,nu] = R
                if mu != nu:
                    self[rho,sigma,nu,mu] = -R