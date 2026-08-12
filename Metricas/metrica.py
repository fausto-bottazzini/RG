from abc import ABC, abstractmethod
import sympy as sp

from Tensores.numerico import CacheNumerico 
from Tensores.cache import TensorCache
from Tensores.tensor import Tensor

from Tensores.derivadas import DerivadasMetrica, DerivadasChristoffel
from Tensores.christoffel import Christoffel
from Tensores.Riemann import Riemann
from Tensores.ricci import Ricci
from Tensores.escalar import EscalarCurvatura

class Metrica(ABC):
    """Clase para cualquier metrica. Coordenadas, parametros y tensor"""

    def __init__(self):
        self.coords = None
        self.params = {}
        self._g = None
        self._g_inv = None
        self._cache = None
        self._numeric_cache = None

    @abstractmethod
    def tensor_metrico(self):
        """Devuelve el tensor de la metrica"""
        pass

    def tensor_metrico_inverso(self):
        """Devuelve la inversa de la métrica si está definida analíticamente."""
        return None

    @property
    def g(self): 
        if self._g is None:
            self._g = self.tensor_metrico()
        return self._g

    @property
    def g_inv(self):
        if self._g_inv is None:
            g_inv = self.tensor_metrico_inverso()

            if g_inv is None:
                g_inv = Tensor.from_matrix(self.g.to_matrix().inv())

            self._g_inv = g_inv
        return self._g_inv
    
    @property
    def cache(self):
        if self._cache is None:
            self._cache = TensorCache(self)
        return self._cache

    @property
    def numeric_cache(self):
        if self._numeric_cache is None:
            self._numeric_cache = CacheNumerico(self)
        return self._numeric_cache

    def derivadas(self):
        """Devuelve las derivadas de la metrica"""
        dg = self.cache.load("derivadas")
        if dg is None:
            dg = DerivadasMetrica(self)
            self.cache.save("derivadas", dg)
        return dg   

    def christoffel(self):
        """Devuelve los simbolos de Christoffel"""
        Gamma = self.cache.load("christoffel")
        if Gamma is None:
            Gamma = Christoffel(self)
            self.cache.save("christoffel", Gamma)
        return Gamma

    def derivadas_christoffel(self):
        """Devuelve las derivadas de los simbolos de Christoffel"""
        dGamma = self.cache.load("derivadas_christoffel")
        if dGamma is None:
            dGamma = DerivadasChristoffel(self)
            self.cache.save("derivadas_christoffel", dGamma)
        return dGamma

    def riemann(self):
        """Devuelve el tensor de Riemann"""
        R = self.cache.load("riemann")
        if R is None:
            R = Riemann(self)
            self.cache.save("riemann", R)
        return R

    def ricci(self):
        """Devuelve el tensor de Ricci"""
        Ric = self.cache.load("ricci")
        if Ric is None:
            Ric = Ricci(self)
            self.cache.save("ricci", Ric)
        return Ric

    def escalar_curvatura(self):
        """Devuelve el escalar de curvatura"""
        R = self.cache.load("escalar_curvatura")
        if R is None:
            R = EscalarCurvatura(self)
            self.cache.save("escalar_curvatura", R)
        return R.value

    def borrar_cache(self):
        """Borra el cache en RAM"""
        self._numeric.clear()
        self.cache.clear()

    def borrar_cache_numerico(self):
        self._numeric.clear()

    def borrar_cache_disco(self):
        """Borra el cache en disco"""
        self.cache.clear_disk()
        self._numeric.clear()

    def numeric(self, name):
        """
        Devuelve una representación numérica cacheada de una cantidad
        tensorial.

        Las expresiones se convierten a funciones NumPy mediante lambdify.
        Se utiliza CSE para reutilizar subexpresiones comunes entre componentes.
        """

        if name in self._numeric:
            return self._numeric[name]

        if name == "metric":
            tensor = self.g

        elif name == "inverse_metric":
            tensor = self.g_inv

        elif name == "christoffel":
            tensor = self.christoffel()

        elif name == "riemann":
            tensor = self.riemann()

        elif name == "ricci":
            tensor = self.ricci()

        else:
            raise ValueError(f"Cantidad numérica desconocida: {name}")

        items = list(tensor.items())

        indices = [idx for idx, _ in items]
        expressions = [expr for _, expr in items]

        function = sp.lambdify(self.coords, expressions, modules="numpy", cse=True)

        def evaluate(*values):
            result = function(*values)
            return {idx: value for idx, value in zip(indices, result)}

        self._numeric[name] = evaluate

        return evaluate

