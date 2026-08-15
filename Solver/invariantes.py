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

class Condiciones:
    """Construcción y validación de condiciones iniciales."""
    TIPOS = {"timelike": -1.0, "null": 0.0, "spacelike": 1.0}

    def __init__(self, metrica):
        self.metrica = metrica
        self.dim = len(metrica.coords)
        self._metric_num = metrica.numeric("metric")
        self._metric_indices = {indice: i for i, indice in enumerate(self._metric_num.indices)}

    def _vector_metric(self, x):
        """Evalua la metrica en las coordenadas x."""
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dim,):
            raise ValueError(f"x debe tener dimension {self.dim}.")

        valores = self._metric_num.evaluar_valores(*x)
        return valores

    def normalizacion(self, x, u):
        """Calcula U^2."""
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)

        if x.shape != (self.dim,):
            raise ValueError(f"x debe tener dimension {self.dim}.")
        if u.shape != (self.dim,):
            raise ValueError(f"u debe tener dimension {self.dim}.")

        valores = self._vector_metric(x)
        norma = 0.0

        for indice, posicion in self._metric_indices.items():
            mu, nu = indice
            norma += valores[posicion] * u[mu] * u[nu]
        return float(norma)

    def validar(self, x, u, tipo = "timelike", atol=1e-10, rtol=1e-10):
        """Comprueba si las condiciones iniciales cumplen la normalización."""
        if tipo not in self.TIPOS:
            raise ValueError(f"Tipo desconocido: {tipo}. Usar uno de {tuple(self.TIPOS)}.")

        objetivo = self.TIPOS[tipo]
        valor = self.normalizacion(x,u)
        error = valor - objetivo
        escala = max(1.0, abs(objetivo))
        cumple = abs(error) <= (atol + rtol * escala)

        return {"cumple": cumple, "tipo": tipo, "objetivo": objetivo, "valor": valor, "error": error}

    def _metric_component(self, valores, mu, nu):
        """Devuelve g mu nu a partir de los valores numéricos sparse."""
        posicion = self._metric_indices.get((mu,nu))
        if posicion is None:
            return 0.0
        return valores[posicion]

    def normalizar(self, x, u, tipo = "timelike", componente=0, signo=None):
        """Ajusta una única componente de la velocidad inicial para satisfacer U^2."""

        if tipo not in self.TIPOS:
            raise ValueError(f"Tipo desconocido: {tipo}. Usar uno de {tuple(self.TIPOS)}.")
        if not 0 <= componente < self.dim:
            raise IndexError("Indice de componente fuera de rango.")

        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float).copy()

        if x.shape != (self.dim,):
            raise ValueError(f"x debe tener dimension {self.dim}.")
        if u.shape != (self.dim,):
            raise ValueError(f"u debe tener dimension {self.dim}.")        

        valores = self._vector_metric(x)
        objetivo = self.TIPOS[tipo]

        a = componente
        A = self._metric_component(valores, a, a)
        
        B = 0.0
        for otro in range(self.dim):
            if otro == a:
                continue
            indice = (a, otro)
            if indice in self._metric_indices:
                posicion = self._metric_indices[indice]
                B += 2.0 * valores[posicion] * u[otro]

        C = 0.0
        for (mu, nu), posicion in self._metric_indices.items():
            if mu == a or nu == a:
                continue
            C += (valores[posicion]* u[mu] * u[nu])

        discriminante = (B**2 - 4.0 * A * (C - objetivo))
        tolerancia = 1e-12 * max(1.0, abs(B**2), abs(4.0 * A * (C - objetivo)))

        if discriminante < -tolerancia:
            raise ValueError("No existe una solucion real para la componente solicitada con las condiciones dadas.")
        discriminante = max(discriminante, 0.0)
        raiz = np.sqrt(discriminante)
        soluciones = ((-B + raiz) / (2.0 * A), (-B - raiz) / (2.0 * A))

        if signo is None:
            candidatos = [valor for valor in soluciones if valor >= 0]
            if not candidatos:
                candidatos = list(soluciones)
            u[a] = candidatos[0]
        else:
            candidatos = [valor for valor in soluciones if np.sign(valor) == np.sign(signo) or np.isclose(valor, 0.0)]
            if not candidatos:
                raise ValueError("No existe una solución con el signo solicitado.")
            u[a] = candidatos[0]

        return u