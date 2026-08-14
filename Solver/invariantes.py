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

    def _momento_covariante(self, g, u):
        return g @ u

    def evaluar(self, y):
        """Evalua los diagnósticos principales. U^2, E, L, Q."""
        y = np.asarray(y, dtype=float)
        coords = y[:self.dim]
        u = y[self.dim:]
        g = self._metric_matrix(coords)
        p = self._momento_covariante(g, u)
        norma = np.dot(p, u)
        energia = -p[0]
        momento_angular = p[3]
        carter = self._carter(coords, p, norma, energia, momento_angular)
        return {"norma": norma, "energia": energia, "momento_angular": momento_angular, "carter": carter}

    def norma(self, y):
        """Calcula u^2"""
        return self.evaluar(y)["norma"]

    def killing(self, y, vector):
        """Cantidad conservada asociada a un vector de killing."""
        y = np.asarray(y, dtype=float)
        coords = y[:self.dim]
        u = y[self.dim:]
        g = self._metric_matrix(coords)
        xi = np.asarray(vector, dtype=float)
        return np.einsum("ij,i,j->", g, xi, u)

    def energia(self, y):
        """Energia asociada al killing temporal."""
        return self.evaluar(y)["energia"]

    def momento_angular(self, y, indice=3):
        """Cantidad conservada asociada a una coordenada cíclica, phi = x^3."""
        y = np.asarray(y, dtype=float)
        coords = y[:self.dim]
        u = y[self.dim:]
        g = self._metric_matrix(coords)
        return np.dot(g[indice], u)

    def _carter(self, coords, p, norma, energia, momento_angular):
        """Constante de Carter para Kerr en coords de Boyer-Lindquist."""
        if "a" not in self.metrica.params:
            return np.nan
        a = float(self.metrica.params["a"])
        theta = coords[2]
        mu2 = -norma
        p_theta = p[2]
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        return (p_theta**2 + cos_theta**2 * (a**2 * (mu2 - energia**2) + momento_angular**2 / sin_theta**2))

    def trayectoria(self, resultado):
        """Evalúa las invariantes sobre toda la trayectoria."""
        estados = resultado.y.T
        n = len(estados)
        norma = np.empty(n, dtype=float)
        energia = np.empty(n, dtype=float)
        momento = np.empty(n, dtype=float)
        carter = np.empty(n, dtype=float)

        for i, estado in enumerate(estados):
            valores = self.evaluar(estado)
            norma[i] = valores["norma"]
            energia[i] = valores["energia"]
            momento[i] = valores["momento_angular"]
            carter[i] = valores["carter"]

        return {"lambda": resultado.t,
                "norma": norma,
                "energia": energia,
                "momento_angular": momento,
                "carter": carter}

    def errores(self, resultado):
        """Calcula la desviación de cada invariante respecto de su valor inicial."""
        datos = self.trayectoria(resultado)
        errores={}
        nombres = ("norma", "energia", "momento_angular", "carter")
        for nombre in nombres:
            valores = datos[nombre]
            inicial = valores[0]
            diferencia = np.abs(valores - inicial)
            escala = max(abs(inicial), np.finfo(float).eps)
            errores[nombre] = {"absoluto": diferencia,
                               "max_absoluto": np.max(diferencia),
                               "relativo": diferencia / escala, 
                               "max_relativo": np.max(diferencia / escala)}
        return errores