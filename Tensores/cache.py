import hashlib
import pickle 
import shutil
from pathlib import Path
import sympy as sp
import config 
from Tensores.tensor import TensorNumerico

class TensorCache:
    def __init__(self, metric, cache_dir=None):
        self.metric = metric
        self.memory = {}
        self.cache_dir = Path(cache_dir) if cache_dir is not None else config.CACHE_DIR
        self.cache_dir.mkdir(exist_ok=True)
        self.metric_id = self._build_metric_hash()
        self.path = self.cache_dir / self.metric_id 
        self.path.mkdir(exist_ok=True)

    def _build_metric_hash(self):
        """Hash para la metrica segun sus parametros y coordenadas"""
        txt = (repr(self.metric.g)
                + repr(self.metric.coords)
                + repr(self.metric.params))
        return hashlib.sha256(txt.encode()).hexdigest()[:16]

    def save(self, name, obj):
        """Guarda un objeto en la cache de la metrica"""
        self.memory[name] = obj
        filename = self.path / f"{name}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(obj, f)

    def load(self, name):
        """Carga un objeto de la cache de la metrica"""
        if name in self.memory:
            return self.memory[name]
        filename = self.path / f"{name}.pkl"
        if filename.exists():
            with open(filename, "rb") as f:
                obj = pickle.load(f)

            if hasattr(obj, "metrica"):
                obj.metrica = self.metric

            self.memory[name] = obj
            return obj
        return None

    def exists(self, name):
        """Verifica si un objeto existe en la cache de la metrica"""
        if name in self.memory:
            return True
        filename = self.path / f"{name}.pkl"
        return filename.exists()

    def clear(self):
        """Borra el cache de la metrica"""
        self.memory.clear()

    def clear_disk(self):
        """Borra completamente el cache de la metrica"""
        self.memory.clear()
        if self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(exist_ok=True)


class CacheNumerico:
    def __init__(self, metric):
        self.metric = metric
        self.memory = {}

    def get(self, name, tensor, modules="numpy"):
        if name in self.memory:
            return self.memory[name]

        funciones = {}
        for idx, expr in tensor.items():
            funciones[idx] = sp.lambdify(self.metric.coords, expr, modules=modules, cse=True)

        numeric_tensor = TensorNumerico(metric=self.metric, rank=tensor.rank, dim=tensor.dim, funciones=funciones)
        self.memory[name] = numeric_tensor
        return numeric_tensor

    def clear(self):
        self.memory.clear()