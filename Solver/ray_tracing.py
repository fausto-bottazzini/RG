import numpy as np
from dataclasses import dataclass
from Solver.geodesica import GeodesicaEvaluator

# estados
STATUS_ACTIVE = 0
STATUS_ESCAPE = 1
STATUS_HORIZON = 2
STATUS_DISK = 3
STATUS_MAX_LAMBDA = 4
STATUS_STEP_FAILURE = 5 

@dataclass
class RayTracingResult:
    estado: np.ndarray
    status: np.ndarray
    parametro: np.ndarray
    pasos_aceptados: np.ndarray
    pasos_rechazados: np.ndarray

class RayEvent:
    """Evento geométrico para la terminación de rayos."""
    code = STATUS_ESCAPE    
    def detect(self, y0, y1):
        raise NotImplementedError

class SurfaceEvent(RayEvent):
    """
    Evento definido por F(y) = 0. 
    direction = 0: cualquier curce
    direction = +1: F pasa de negativo a positivo
    direction = -1: F pasa de positivo a negativo
    """
    def __init__(self, *, direction=0):
        self.direction = direction

    def value(self, y):
        raise NotImplementedError

    def detect(self, y0, y1):
        f0 = np.asarray(self.value(y0), dtype=float)
        f1 = np.asarray(self.value(y1), dtype=float)

        if self.direction > 0:
            mask = ((f0 < 0.0) & (f1 >= 0.0))
        elif self.direction < 0:
            mask = ((f0 > 0.0) & (f1 <= 0.0))
        else:
            mask = ((f0 < 0.0) & (f1 >= 0.0)) | ((f0 > 0.0) & (f1 <= 0.0))

        denominator = f0 - f1
        alpha = np.zeros_like(f0)
        np.divide(f0, denominator, out=alpha, where=np.abs(denominator) > np.finfo(float).eps)
        alpha = np.clip(alpha, 0.0, 1.0)
        return mask, alpha

class EscapeEvent(SurfaceEvent):
    """Termina el rayo cuando r >= r_max."""
    code = STATUS_ESCAPE
    def __init__(self, r_max, radial_index=1):
        super().__init__(direction=+1)
        self.r_max = float(r_max)
        self.radial_index = radial_index

    def value(self, y):
        return y[:, self.radial_index] - self.r_max

class HorizonEvent(SurfaceEvent): # Event-Horizon xd
    """Termina el rayo cuando r <= r_stop."""
    code = STATUS_HORIZON
    def __init__(self, r_stop, radial_index=1):
        super().__init__(direction=-1)
        self.r_stop = float(r_stop)
        self.radial_index = radial_index

    def value(self, y):
        return y[:, self.radial_index] - self.r_stop

class DiskEvent(SurfaceEvent):
    """Intersección con un disco ecuatorial (theta_disk).
    La intersección solo cuenta dentro de [r_in, r_out]."""
    code = STATUS_DISK
    def __init__(self, r_in, r_out, *, theta_disk = np.pi/2.0, radial_index=1, theta_index=2):
        super().__init__(direction=0)
        self.r_in = float(r_in)
        self.r_out = float(r_out)
        self.theta_disk = float(theta_disk)
        self.radial_index = radial_index
        self.theta_index = theta_index

    def value(self, y):
        return y[:, self.theta_index] - self.theta_disk

    def detect(self, y0, y1):
        mask, alpha = super().detect(y0, y1)
        y_hit = (y0 + alpha[:, None] * (y1-y0))
        r_hit = y_hit[:, self.radial_index]
        mask &= ((r_hit >= self.r_in)) & (r_hit <= self.r_out)
        return mask, alpha 

class SolverRayTracing:
    """Integrados batch para geodésicas nulas."""
    def __init__(self, metrica, *, rtol=1e-7, atol=1e-9, safety=0.9, min_factor=0.2, max_factor=5.0): 
        self.evaluator = GeodesicaEvaluator(metrica)
        self.dim = self.evaluator.dim
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.safety = float(safety)
        self.min_factor = float(min_factor)
        self.max_factor = float(max_factor)

    def rhs(self, y):
        y = np.asarray(y, dtype=float)
        if y.ndim != 2:
            raise ValueError("y debe tener la forma (N, 2*dim).")
        if y.shape[1] != 2*self.dim:
            raise ValueError(f"y debe tener {2*self.dim} columnas.")

        x = y[:, :self.dim]
        u = y[:, self.dim:]

        acc = self.evaluator._acceleration_function(
            *[x[:,i] for i in self.evaluator._coordinate_indices],
            *[u[:,i] for i in self.evaluator._velocity_indices],
        )

        out = np.empty_like(y)
        out[:, :self.dim] = u
        out[:, self.dim:] = np.column_stack(acc)

        return out

    def step_rk4(self, y, h):
        k1 = self.rhs(y)
        k2 = self.rhs(y + 0.5 * h * k1)
        k3 = self.rhs(y + 0.5 * h * k2)
        k4 = self.rhs(y + h * k3)
        return y + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

    def _rk45(self, y, h): # Dormand-Prince 5(4)
        h = np.asarray(h, dtype=float)
        k1 = self.rhs(y)
        k2 = self.rhs(y + h[:, None] * (1.0 / 5.0 * k1))
        k3 = self.rhs(y + h[:, None] * (3.0 / 40.0 * k1 + 9.0 / 40.0 * k2))
        k4 = self.rhs(y + h[:, None] * (44.0 / 45.0 * k1 - 56.0 / 15.0 * k2 + 32.0 / 9.0 * k3))
        k5 = self.rhs(y + h[:, None] * (19372.0 / 6561.0 * k1 - 25360.0 / 2187.0 * k2 + 64448.0 / 6561.0 * k3 - 212.0 / 729.0 * k4))
        k6 = self.rhs(y + h[:, None] * (9017.0 / 3168.0 * k1 - 355.0 / 33.0 * k2 + 46732.0 / 5247.0 * k3 + 49.0 / 176.0 * k4 - 5103.0 / 18656.0 * k5))
        k7 = self.rhs(y + h[:, None] * (35.0 / 384.0 * k1 + 500.0 / 1113.0 * k3 + 125.0 / 192.0 * k4 - 2187.0 / 6784.0 * k5 + 11.0 / 84.0 * k6))

        y5 = y + h[:, None] * (35.0 / 384.0 * k1 + 500.0 / 1113.0 * k3 + 125.0 / 192.0 * k4 - 2187.0 / 6784.0 * k5 + 11.0 / 84.0 * k6) 
        y4 = y + h[:, None] * (5179.0 / 57600.0 * k1 + 7571.0 / 16695.0 * k3 + 393.0 / 640.0 * k4 - 92097.0 / 339200.0 * k5 + 187.0 / 2100.0 * k6 + 1.0 / 40.0 * k7)

        error = y5 - y4
        scale = (self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y5)))
        error_norm = np.max(np.abs(error) / scale, axis=1)
        return y5, error_norm

    def _factor_paso(self, error, aceptado):
        factor = np.empty_like(error)
        cero = error == 0.0
        factor[cero] = self.max_factor
        no_cero = ~cero
        factor[no_cero] = (self.safety * error[no_cero] ** (-1.0 / 5.0))

        if np.any(aceptado):
            factor[aceptado] = np.clip(factor[aceptado], self.min_factor, self.max_factor)

        rechazado = ~aceptado
        if np.any(rechazado):
            factor[rechazado] = np.clip(factor[rechazado], 0.1, 0.5)

        return factor

    def resolver(self, y0, *, lambda_max, h0=0.01, h_min=1e-8, h_max=np.inf, max_steps=1_000_000):
        y = np.asarray(y0, dtype=float).copy()

        if y.ndim != 2:
            raise ValueError("y0 debe tener forma (N, 2*dim).")
        if y.shape[1] != 2 * self.dim:
            raise ValueError(f"y0 debe tener {2 * self.dim} columnas.")

        N = len(y)
        parametro = np.zeros(N)
        h = np.full(N, h0)
        pasos_aceptados = np.zeros(N, dtype=int)
        pasos_rechazados = np.zeros(N, dtype=int)

        for _ in range(max_steps):
            activos = parametro < lambda_max
            if not np.any(activos):
                break

            indices = np.flatnonzero(activos)
            y_act = y[indices]
            h_act = h[indices]

            restante = lambda_max - parametro[indices]

            h_act = np.minimum(h_act, restante)
            y_new, error = self._rk45(y_act, h_act)

            aceptado = error <= 1.0
            if np.any(~aceptado):
                idx = indices[~aceptado]
                factor = self._factor_paso(error[~aceptado], np.zeros(np.sum(~aceptado), dtype=bool))
                h[idx] *= factor
                h[idx] = np.maximum(h[idx], h_min)
                pasos_rechazados[idx] += 1

            if np.any(aceptado):
                idx = indices[aceptado]
                y[idx] = y_new[aceptado]
                parametro[idx] += h_act[aceptado]
                pasos_aceptados[idx] += 1
                factor = self._factor_paso(error[aceptado], np.ones(np.sum(aceptado), dtype=bool))
                h[idx] *= factor
                h[idx] = np.clip(h[idx], h_min, h_max)

        return RayTracingResult(
            estado=y,
            status=np.where(parametro >= lambda_max, STATUS_MAX_LAMBDA, STATUS_ACTIVE),
            parametro=parametro,
            pasos_aceptados=pasos_aceptados,
            pasos_rechazados=pasos_rechazados,
        )   

