import numpy as np

class Invariantes:
    """
    Cantidades conservadas y diagnóstico de geodesica.
        y = (x^mu, u^mu)
    """

    def __init__(self, metrica):
        self.metrica = metrica
        self.dim = metrica.dim

    def norma(self, y):
        """Calcula u^2"""
        y = np.asarray(y, dtype=float)
        u = y[self.dim:]
        g = self.metrica.evaluar(*y[:self.dim])
        return np.einsum("ij,i,j->", g, u, u)

    def killing(self, y, covector):
        """Cantidad conservada asociada al covector de killing."""
        y = np.asarray(y, dtype=float)
        u = y[self.dim:]
        g = self.metrica.evaluar(*y[:self.dim])
        xi = np.asarray(covector, dtype=float)
        return np.einsum("ij,i,j->", g, xi, u)

    def energia(self, y):
        """Energia asociada al killing temporal."""
        xi = np.zeros(self.dim, dtype=float)
        xi[0] = 1.0
        return -self.killing(y, xi)

    def momento_angular(self, y, indice=3):
        """Cantidad conservada asociada a una coordenada angular, phi = x^3."""
        xi = np.zeros(self.dim, dtype=float)
        xi[indice] = 1.0
        return self.killing(y, xi)

    def evaluar(self, y):
        """Devuelve los diagnósticos principales."""
        return {"norma": self.norma(y), "energia": self.energia(y), "momento_angular": self.momento_angular(y)}

    def trayectoria(self, resultado):
        """Evalúa las invariantes sobre toda la trayectoria."""
        estados = resultado.y.T
        norma = np.empty(len(estados), dtype=float)
        energia = np.empty(len(estados), dtype=float)
        momento = np.empty(len(estados), dtype=float)

        for i, estado in enumerate(estados):
            norma[i] = self.norma(estado)
            energia[i] = self.energia(estado)
            momento[i] = self.momento_angular(estado)

        return {"lambda": resultado.t,
                "norma": norma,
                "energia": energia,
                "momento_angular": momento}