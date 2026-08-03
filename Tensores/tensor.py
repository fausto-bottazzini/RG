import sympy as sp

class Tensor:
    """Clase para cualquier tensor. Administra almacenamiento"""

    def __init__(self, rank, dim):
        self.rank = rank
        self.dim = dim
        self.index = [{} for _ in range(rank)]
        self.data = {}

    def __getitem__(self, index):
        return self.data.get(index, sp.Integer(0))

    def __setitem__(self, index, value):
        self.data[index] = value
        for axis, idx in enumerate(index):
            self.index[axis].setdefault(idx, []).append(index)

    def by_index(self, axis, value):
        return self.index[axis].get(value, [])

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

    def __repr__(self):
        return (f"{self.__class__.__name__}"
                f"(rank={self.rank}, dim={self.dim}, nnz={self.nnz})")