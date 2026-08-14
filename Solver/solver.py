import numpy as np
from scipy.integrate import solve_ivp
from Solver.geodesica import GeodesicaEvaluator

class SolverGeodesica:
    def __init__(self, metrica):
        self.evaluator = GeodesicaEvaluator(metrica)
        self.dim = self.evaluator.dim

    def _rhs(self, lam, y):
        out = np.empty(2 * self.dim, dtype = float)
        self.evaluator.sistema(y, out)
        return out

    def resolver(self, x0, u0, intervalo, *, metodo="DOP853", rtol=1e-9, atol=1e-11, max_step=np.inf):
        """
        Integra una geodésica.

        Parámetros
        ----------
        x0 : array_like
            Posición inicial.

        u0 : array_like
            Velocidad inicial.

        intervalo : tuple
            (lambda_inicial, lambda_final)
        """

        x0 = np.asarray(x0, dtype=float)
        u0 = np.asarray(u0, dtype=float)

        if x0.shape != (self.dim,):
            raise ValueError(f"x0 debe tener forma ({self.dim},)")

        if u0.shape != (self.dim,):
            raise ValueError(f"u0 debe tener forma ({self.dim},)")

        y0 = np.empty(2 * self.dim, dtype=float)

        y0[:self.dim] = x0
        y0[self.dim:] = u0

        return solve_ivp(self._rhs, intervalo, y0, method=metodo, rtol=rtol, atol=atol, max_step=max_step)

