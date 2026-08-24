import numpy as np

class Background:
    """Interfaz para un fondo muestreable por posición/dirección."""

    def sample(self, position, direction):
        """
        Parameters
        ----------
        position : ndarray, shape (N, 3)
            Punto de intersección del rayo con la superficie/fondo.

        direction : ndarray, shape (N, 3)
            Dirección de propagación del rayo.

        Returns
        -------
        colors : ndarray, shape (N, 3)
            RGB en [0, 1].
        """
        raise NotImplementedError


class GridBackground(Background):
    """
    Fondo plano cuadriculado.

    El fondo se considera el plano x = -distance.
    Las coordenadas (y, z) sobre ese plano generan la cuadrícula.
    """

    def __init__(self, *, distance=100.0,
                extent=60.0, spacing=10.0,
                linewidth=1.0, line_value=0.9,
                background_value=0.04,
                subgrid=True, subspacing=None, subline_value=0.25,
                tolerance=0.035):
        
        self.distance = float(distance)
        self.extent = float(extent)
        self.spacing = float(spacing)
        self.linewidth = float(linewidth)
        self.line_value = float(line_value)
        self.background_value = float(background_value)

        self.subgrid = bool(subgrid)

        if subspacing is None:
            subspacing = self.spacing / 5.0

        self.subspacing = float(subspacing)
        self.subline_value = float(subline_value)

        self.tolerance = float(tolerance)

        if self.distance <= 0:
            raise ValueError("distance debe ser positivo.")

        if self.extent <= 0:
            raise ValueError("extent debe ser positivo.")

        if self.spacing <= 0:
            raise ValueError("spacing debe ser positivo.")

        if self.subspacing <= 0:
            raise ValueError("subspacing debe ser positivo.")

    @staticmethod
    def _near_grid(value, spacing, tolerance):
        """Detecta proximidad a una línea periódica."""
        q = np.mod(value + 0.5 * spacing, spacing)
        distance = np.minimum(q, spacing - q)
        return distance <= tolerance

    def sample(self, position, direction):
        position = np.asarray(position, dtype=float)

        if position.ndim != 2 or position.shape[1] != 3:
            raise ValueError("position debe tener forma (N, 3).")

        y = position[:, 1]
        z = position[:, 2]

        inside = ((np.abs(y) <= self.extent) & (np.abs(z) <= self.extent))
        values = np.full(len(position), self.background_value, dtype=float)

        grid_y = self._near_grid(y, self.spacing, self.tolerance * self.spacing)
        grid_z = self._near_grid(z, self.spacing, self.tolerance * self.spacing)

        principal = (inside & (grid_y | grid_z))
        values[principal] = self.line_value

        if self.subgrid:
            sub_y = self._near_grid(y, self.subspacing, self.tolerance * self.subspacing)
            sub_z = self._near_grid(z, self.subspacing, self.tolerance * self.subspacing)

            sub = (inside & (sub_y | sub_z) & ~principal)
            values[sub] = self.subline_value

        return np.repeat(values[:, None], 3, axis=1)