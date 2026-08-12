import sympy as sp

class Tensor:
    """Clase para cualquier tensor simbólico. Administra almacenamiento"""

    def __init__(self, rank, dim):
        self.rank = rank
        self.dim = dim
        self.index = [{} for _ in range(rank)]
        self.data = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("metrica", None)
        return state

    def __getitem__(self, index):
        return self.data.get(index, sp.Integer(0))

    def __setitem__(self, index, value):
        if value == 0:
            self._remove(index)
            return
        if index in self.data:
            self.data[index] = value
            return

        self.data[index] = value
        for axis, idx in enumerate(index):
            self.index[axis].setdefault(idx, set()).add(index)

    def _remove(self, index):
        if index not in self.data:
            return
        
        del self.data[index]
        for axis, idx in enumerate(index):
            candidates = self.index[axis].get(idx)
            
            if candidates is None:
                continue

            candidates.discard(index)
            if not candidates:
                del self.index[axis][idx]

    def by_index(self, axis, value):
        return self.index[axis].get(value, set())

    def select(self, filters=None):
        if filters is None:
            yield from self.items()
            return

        axis, value = min(filters.items(), key=lambda item: len(self.by_index(*item)))
        candidates = self.by_index(axis, value)

        for idx in candidates:
            if all(idx[a] == v for a, v in filters.items()):
                yield idx, self.data[idx]

    def __contains__(self, index):
        return index in self.data

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        return (self.dim,) * self.rank

    @property
    def nnz(self):
        return len(self.data)

    @classmethod
    def from_matrix(cls, matrix):
        dim = matrix.rows
        T = cls(rank = 2, dim = dim)
        for i in range(dim):
            for j in range(dim):
                if matrix[i, j] != 0:
                    T[i, j] = matrix[i, j]
        return T

    def to_matrix(self):
        M = sp.zeros(self.dim)
        for idx, value in self.items():
            M[idx] = value 
        return M

    def clear(self):
        self.data.clear()
        self.index = [{} for _ in range(self.rank)]

    def __repr__(self):
        return (f"{self.__class__.__name__}"
                f"(rank={self.rank}, dim={self.dim}, nnz={self.nnz})")


class TensorNumerico:
    """
    Representación numérica de un tensor sparse.

    Cada componente se almacena como una función numérica
    obtenida mediante lambdify.
    """

    def __init__(self, metric, rank, dim, funciones):
        self.metrica = metric
        self.rank = rank
        self.dim = dim
        self._data = funciones

    def __getitem__(self, idx):
        return self._data[idx]

    def __iter__(self):
        return iter(self._data)

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def __len__(self):
        return len(self._data)

    @property
    def indices(self):
        return self._data.keys()

    def evaluar_componente(self, idx, *coords):
        """Evalúa un único componente en las coordenadas dadas."""
        return self._data[idx](*coords)

    def evaluar(self, *coords):
        """
        Evalúa todos los componentes almacenados.

        Devuelve un diccionario:
            índice -> valor numérico
        """
        return {idx: funcion(*coords) for idx, funcion in self._data.items()}