from abc import ABC, abstractmethod
import sympy as sp

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

    @abstractmethod
    def tensor_metrico(self):
        """Devuelve el tensor de la metrica"""
        pass

    @property
    def g(self): 
        if self._g is None:
            self._g = self.tensor_metrico()
        return self._g

    @property
    def g_inv(self):
        if self._g_inv is None:
            self._g_inv = Tensor.from_matrix(self.g.to_matrix().inv())
        return self._g_inv

    @property
    def cache(self):
        if self._cache is None:
            self._cache = TensorCache(self)
        return self._cache

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
        self.cache.clear()

    def borrar_cache_disco(self):
        """Borra el cache en disco"""
        self.cache.clear_disk()

