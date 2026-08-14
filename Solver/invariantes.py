import numpy as np

class Invariantes:
    """
    Cantidades conservadas y diagnóstico de geodesica.
        y = (x^mu, u^mu)
    """

    def __init__(self, metrica):
        self.metrica = metrica
        self.dim = metrica.g.dim
        self._metric_num = metrica.numeric("metric")

    def _metric_matrix(self, coords):
        """Evalua la metrica y reconstruye la matriz densa."""
        valores = self._metric_num.evaluar_valores(*coords)
        g = np.zeros((self.dim, self.dim), dtype=float)
        for idx, valor in zip(self._metric_num.indices, valores):
            g[idx] = valor
        return g

    def norma(self, y):
        """Calcula u^2"""
        y = np.asarray(y, dtype=float)
        coords = y[:self.dim]
        u = y[self.dim:]
        g = self._metric_matrix(coords)
        return np.einsum("ij,i,j->", g, u, u)

    def killing(self, y, vector):
        """Cantidad conservada asociada al vector de killing."""
        y = np.asarray(y, dtype=float)
        coords = y[:self.dim]
        u = y[self.dim:]
        g = self._metric_matrix(coords)
        xi = np.asarray(vector, dtype=float)
        return np.einsum("ij,i,j->", g, xi, u)

    def energia(self, y):
        """Energia asociada al killing temporal."""
        y = np.asarray(y, dtype=float)
        coords = y[:self.dim]
        u = y[self.dim:]
        g = self._metric_matrix(coords)
        return -np.dot(g[0], u)

    def momento_angular(self, y, indice=3):
        """Cantidad conservada asociada a una coordenada cíclica, phi = x^3."""
        y = np.asarray(y, dtype=float)
        coords = y[:self.dim]
        u = y[self.dim:]
        g = self._metric_matrix(coords)
        return np.dot(g[indice], u)

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
            valores = self.evaluar(estado)
            norma[i] = valores["norma"]
            energia[i] = valores["energia"]
            momento[i] = valores["momento_angular"]

        return {"lambda": resultado.t,
                "norma": norma,
                "energia": energia,
                "momento_angular": momento}

    def errores(self, resultado):
        """Calcula la desviación de cada invariante respecto de su valor inicial."""
        datos = self.trayectoria(resultado)
        errores={}
        for nombre in ("norma", "energia", "momento_angular"):
            valores = datos[nombre]
            inicial = valores[0]
            errores[nombre] = {"absoluto": np.abs(valores - inicial),
                               "max_absoluto": np.max(np.abs(valores - inicial))}
            escala = max(abs(inicial), np.finfo(float).eps)
            errores[nombre]["relativo"] = (np.abs(valores - inicial) / escala)
            errores[nombre]["max_relativo"] = np.max(errores[nombre]["relativo"])
        return errores