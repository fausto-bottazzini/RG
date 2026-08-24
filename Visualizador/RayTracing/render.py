import numpy as np
from Solver.ray_tracing import (STATUS_ESCAPE, STATUS_HORIZON)

class RayTracingRenderer:
    """Convierte el resultado del ray tracing en una imagen."""
    def __init__(self, background, *, horizon_color=(0.0, 0.0, 0.0)):
        self.background = background
        self.horizon_color = np.asarray(horizon_color, dtype=float)

        if self.horizon_color.shape != (3,):
            raise ValueError("horizon_color debe tener forma (3,).")

    @staticmethod
    def _spherical_to_cartesian(state):
        """
        Convierte una colección de estados
            (t, r, theta, phi,
             ut, ur, utheta, uphi)
        al punto cartesiano y dirección cartesiana.
        """
        r = state[:, 1]
        theta = state[:, 2]
        phi = state[:, 3]

        ur = state[:, 5]
        utheta = state[:, 6]
        uphi = state[:, 7]

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        x = (r * sin_theta * cos_phi)
        y = (r * sin_theta * sin_phi)
        z = (r * cos_theta)

        position = np.column_stack((x, y, z))

        vx = (ur * sin_theta * cos_phi + r * utheta * cos_theta * cos_phi - r * sin_theta * uphi * sin_phi)
        vy = (ur * sin_theta * sin_phi + r * utheta * cos_theta * sin_phi + r * sin_theta * uphi * cos_phi)
        vz = (ur * cos_theta - r * utheta * sin_theta)

        direction = np.column_stack((vx, vy, vz))
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        direction /= np.maximum(norm, np.finfo(float).eps)

        return position, direction

    @staticmethod
    def _intersect_plane_x(position, direction, x_plane):
        """
        Intersección de una recta:
            p + s d
        con:
            x = x_plane
        Devuelve el punto de intersección y una máscara
        para rayos que realmente llegan al plano.
        """
        dx = direction[:, 0]
        numerador = (x_plane - position[:, 0])
        valid_direction = (np.abs(dx) > np.finfo(float).eps)

        s = np.zeros(len(position), dtype=float)
        np.divide(numerador, dx, out=s, where=valid_direction)

        valid = (valid_direction & (s > 0.0))
        hit = (position + s[:, None] * direction)

        return hit, valid

    def render(self, resultado, camara, *, background_distance=None):
        """Genera una imagen RGB segun los estados."""
        if not hasattr(resultado, "estado"):
            raise ValueError("resultado debe ser un RayTracingResult.")

        estado = np.asarray(resultado.estado, dtype=float)
        status = np.asarray(resultado.status)

        N = len(estado)
        if N != camara.width * camara.height:
            raise ValueError("La cantidad de rayos no coincide con la resolución de la cámara.")
        if background_distance is None:
            background_distance = (self.background.distance)

        image = np.zeros((N, 3), dtype=float)

        escape = (status == STATUS_ESCAPE)
        if np.any(escape):
            position, direction = (self._spherical_to_cartesian(estado[escape]))
            hit, valid = (self._intersect_plane_x(position, direction, -float(background_distance)))
            indices = np.flatnonzero(escape)

            if np.any(valid):
                colors = self.background.sample(hit[valid], direction[valid])
                image[indices[valid]] = colors

        horizon = (status == STATUS_HORIZON)
        if np.any(horizon):
            image[horizon] = (self.horizon_color)

        image = np.clip(image, 0.0, 1.0)

        return image.reshape(camara.shape + (3,))