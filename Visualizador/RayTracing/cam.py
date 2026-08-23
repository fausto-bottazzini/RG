import numpy as np

class Camara:
    def __init__(self, resolucion, fov, posicion=(0.0, 0.0, 0.0), foward=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0)):
        self.width, self.height = map(int, resolucion)
        self.fov = float(fov)

        self.posicion = np.asarray(posicion, dtype=float)
        self.foward = self._normalize(foward)
        self.up = self._normalize(up)

        self._build_basis()

    @staticmethod
    def _normalize(v):
        v = np.asarray(v, dtype=float)
        norm = np.linalg.norm(v)

        if norm == 0.0:
            raise ValueError("No se puede normalizar un vector nulo.")
        
        return v/norm

    @property
    def shape(self):
        return self.height, self.width

    def _build_basis(self):
        self.right = np.cross(self.foward, self.up)
        norm = np.linalg.norm(self.right)

        if norm == 0.0:
            raise ValueError("Foward y up no pueden ser paralelos.")

        self.right /= norm
        self.up = np.cross(self.right, self.foward)

    def rays_directions(self):
        aspect = self.width / self.height
        half_height = np.tan(self.fov / 2.0)
        half_width = aspect * half_height

        x = np.linspace(-half_width, half_width, self.width)
        y = np.linspace(-half_height, half_height, self.height)
        X, Y = np.meshgrid(x,y)

        directions = (self.foward + X[..., None] * self.right + Y[..., None] * self.up)
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

        return directions

    def rays(self):
        return self.rays_directions().reshape(-1,3)

    def rays_local(self):
        """Genera vectores nulos iniciales en la tetrada local."""
        directions = self.rays()
        return np.column_stack((-np.ones(len(directions)), directions))




