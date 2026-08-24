import numpy as np
from dataclasses import dataclass
from Solver.geodesica import GeodesicaEvaluator
import time

# estados
STATUS_ACTIVE = 0
STATUS_ESCAPE = 1
STATUS_HORIZON = 2
STATUS_DISK = 3
STATUS_MAX_LAMBDA = 4
STATUS_STEP_FAILURE = 5 

EPS = np.finfo(float).eps

@dataclass
class RayTracingResult:
    estado: np.ndarray
    status: np.ndarray
    parametro: np.ndarray
    pasos_aceptados: np.ndarray
    pasos_rechazados: np.ndarray

# eventos

class RayEvent:
    """Evento geométrico que puede terminar un rayo."""
    code = STATUS_ESCAPE
    def detect(self, y0, y1):
        raise NotImplementedError

class SurfaceEvent(RayEvent):
    """
    Evento definido por una superficie F(y) = 0.
    direction = 0:cualquier cruce.
    direction = +1: F pasa de negativo a positivo.
    direction = -1: F pasa de positivo a negativo.
    """
    def __init__(self, *, direction=0):
        if direction not in (-1, 0, +1):
            raise ValueError("direction debe ser -1, 0 o +1.")
        self.direction = direction

    def value(self, y):
        raise NotImplementedError

    def detect(self, y0, y1):
        f0 = np.asarray(self.value(y0), dtype=float)
        f1 = np.asarray(self.value(y1), dtype=float)

        if self.direction > 0:
            detectado = ((f0 < 0.0) & (f1 >= 0.0))
        elif self.direction < 0:
            detectado = ((f0 > 0.0) & (f1 <= 0.0))
        else:
            detectado = (((f0 < 0.0) & (f1 >= 0.0)) | ((f0 > 0.0) & (f1 <= 0.0)))

        denominador = f1 - f0
        alpha = np.zeros_like(f0)
        np.divide(-f0, denominador, out=alpha, where=np.abs(denominador) > EPS)
        alpha = np.clip(alpha, 0.0, 1.0)
        return detectado, alpha

class EscapeEvent(SurfaceEvent):
    """Termina el rayo cuando r alcanza r_max."""
    code = STATUS_ESCAPE
    def __init__(self, r_max, *, radial_index=1):
        super().__init__(direction=+1)
        self.r_max = float(r_max)
        self.radial_index = int(radial_index)

    def value(self, y):
        return (y[:, self.radial_index] - self.r_max)

class HorizonEvent(SurfaceEvent):
    """Termina el rayo cuando alcanza r_stop."""
    code = STATUS_HORIZON
    def __init__(self, r_stop, *, radial_index=1):
        super().__init__(direction=-1)
        self.r_stop = float(r_stop)
        self.radial_index = int(radial_index)

    def value(self, y):
        return (y[:, self.radial_index] - self.r_stop)

class DiskEvent(SurfaceEvent):
    """
    Detecta el cruce de un disco ecuatorial (theta_disk)
        r_in <= r_hit <= r_out
    """
    code = STATUS_DISK
    def __init__(self, r_in, r_out, *, theta_disk=np.pi / 2.0, radial_index=1, theta_index=2):
        super().__init__(direction=0)
        self.r_in = float(r_in)
        self.r_out = float(r_out)
        self.theta_disk = float(theta_disk)
        self.radial_index = int(radial_index)
        self.theta_index = int(theta_index)

    def value(self, y):
        return (y[:, self.theta_index] - self.theta_disk)

    def detect(self, y0, y1):
        detectado, alpha = super().detect(y0, y1)
        y_hit = (y0 + alpha[:, None] * (y1 - y0))
        r_hit = y_hit[:, self.radial_index]
        dentro = ((r_hit >= self.r_in) & (r_hit <= self.r_out))
        detectado &= dentro
        return detectado, alpha 

# solver

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
        self._coordinate_indices = (self.evaluator._coordinate_indices)
        self._velocity_indices = (self.evaluator._velocity_indices)
        self._buffers = None

    def _ensure_buffers(self, capacity):
        if (self._buffers is not None and self._buffers["capacity"] >= capacity):
            return
        
        shape = (capacity, 2*self.dim)
        self._buffers = {
            "capacity": capacity,
            "k1": np.empty(shape),
            "k2": np.empty(shape),
            "k3": np.empty(shape),
            "k4": np.empty(shape),
            "k5": np.empty(shape),
            "k6": np.empty(shape),
            "k7": np.empty(shape),
            "stage": np.empty(shape),
            "y5": np.empty(shape),
            "y4": np.empty(shape),
            "scale": np.empty(shape),
            "error_norm": np.empty(capacity, dtype=float),
            "factor": np.empty(capacity, dtype=float),
            "h_trial": np.empty(capacity, dtype=float),
            "accepted": np.empty(capacity, dtype=bool),
            "rejected": np.empty(capacity, dtype=bool),
        }

    def rhs(self, y, out=None):
        y = np.asarray(y, dtype=float)
        if y.ndim != 2:
            raise ValueError("y debe tener la forma (N, 2*dim).")
        if y.shape[1] != 2*self.dim:
            raise ValueError(f"y debe tener {2*self.dim} columnas.")

        if out is None:
            out = np.empty_like(y)

        x = y[:, :self.dim]
        u = y[:, self.dim:]

        args = (tuple(x[:, i] for i in self._coordinate_indices) + tuple(u[:, i] for i in self._velocity_indices))
        acc = self.evaluator._acceleration_function(*args)

        out[:, :self.dim] = u
        for i, value in enumerate(acc):
            out[:, self.dim + i] = value
        return out

    def step_rk4(self, y, h):
        k1 = self.rhs(y)
        k2 = self.rhs(y + 0.5 * h * k1)
        k3 = self.rhs(y + 0.5 * h * k2)
        k4 = self.rhs(y + h * k3)
        return y + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

    def _rk45(self, y, h): # Dormand-Prince 5(4)
        n = len(y)
        self._ensure_buffers(n)
        b = self._buffers

        k1 = b["k1"][:n]
        k2 = b["k2"][:n]
        k3 = b["k3"][:n]
        k4 = b["k4"][:n]
        k5 = b["k5"][:n]
        k6 = b["k6"][:n]
        k7 = b["k7"][:n]

        stage = b["stage"][:n]

        y5 = b["y5"][:n]
        y4 = b["y4"][:n]

        scale = b["scale"][:n]
        error_norm = b["error_norm"][:n]

        h = np.asarray(h, dtype=float)

        h_col = h[:, None]

        self.rhs(y, out=k1)

        np.multiply(k1, 1.0 / 5.0, out=stage)
        stage *= h_col
        stage += y

        self.rhs(stage, out=k2)

        np.multiply(k1, 3.0 / 40.0, out=stage)
        stage += (9.0 / 40.0) * k2
        stage *= h_col
        stage += y

        self.rhs(stage, out=k3)

        np.multiply(k1, 44.0 / 45.0, out=stage)
        stage += (-56.0 / 15.0) * k2
        stage += (32.0 / 9.0) * k3
        stage *= h_col
        stage += y

        self.rhs(stage, out=k4)

        np.multiply(k1, 19372.0 / 6561.0, out=stage)
        stage += (-25360.0 / 2187.0) * k2
        stage += (64448.0 / 6561.0) * k3
        stage += (-212.0 / 729.0) * k4
        stage *= h_col
        stage += y

        self.rhs(stage, out=k5)

        np.multiply(k1, 9017.0 / 3168.0, out=stage)
        stage += (-355.0 / 33.0) * k2
        stage += (46732.0 / 5247.0) * k3
        stage += (49.0 / 176.0) * k4
        stage += (-5103.0 / 18656.0) * k5
        stage *= h_col
        stage += y

        self.rhs(stage, out=k6)

        np.multiply(k1, 35.0 / 384.0, out=stage)
        stage += (500.0 / 1113.0) * k3
        stage += (125.0 / 192.0) * k4
        stage += (-2187.0 / 6784.0) * k5
        stage += (11.0 / 84.0) * k6
        stage *= h_col
        stage += y
    
        self.rhs(stage, out=k7)

        np.multiply(k1, 35.0 / 384.0, out=y5)
        np.add(y5, (500.0 / 1113.0) * k3, out=y5)
        np.add(y5, (125.0 / 192.0) * k4, out=y5)
        np.add(y5, (-2187.0 / 6784.0) * k5, out=y5)
        np.add(y5, (11.0 / 84.0) * k6, out=y5)

        y5 *= h_col
        y5 += y

        np.multiply(k1, 5179.0 / 57600.0, out=y4)
        np.add(y4, (7571.0 / 16695.0) * k3, out=y4)
        np.add(y4, (393.0 / 640.0) * k4, out=y4)
        np.add(y4, (-92097.0 / 339200.0) * k5, out=y4)
        np.add(y4, (187.0 / 2100.0) * k6, out=y4)
        np.add(y4, (1.0 / 40.0) * k7, out=y4)

        y4 *= h_col
        y4 += y

        y4 -= y5
        np.abs(y4, out=y4)
        np.abs(y5, out=scale)
        np.abs(y, out=stage)
        np.maximum(scale, stage, out=scale)

        scale *= self.rtol
        scale += self.atol

        np.divide(y4, scale, out=y4)
        np.max(y4, axis=1, out=error_norm)

        return (y5, error_norm)

    def _factor_paso(self, error, aceptado, out=None):
        if out is None:
            out = np.empty_like(error)

        cero = error == 0.0
        out[cero] = self.max_factor
        no_cero = ~cero
        out[no_cero] = (self.safety * error[no_cero] ** (-1.0 / 5.0))

        if np.any(aceptado):
            out[aceptado] = np.clip(out[aceptado], self.min_factor, self.max_factor)

        rechazado = ~aceptado
        if np.any(rechazado):
            out[rechazado] = np.clip(out[rechazado], 0.1, 0.5)

        return out

    def _lambda_max_auto(self, y0, eventos, factor=10.0):
        r0 = np.max(y0[:,1])
        r_max = None
        for evento in eventos:
            if isinstance(evento, EscapeEvent):
                r_max = evento.r_max
                break
        if r_max is None:
            raise ValueError("lambda_max = None requiere un EscapeEvent para calcular automáticamente el límite.")
        return factor * (r0 + r_max)

    def resolver(self, y0, *, lambda_max=None, h0=0.01, h_min=1e-8, h_max=np.inf, eventos=(), max_steps=1_000_000, progress=True, progress_interval=1.0):
        y = np.asarray(y0, dtype=float)

        if y.ndim != 2:
            raise ValueError("y0 debe tener forma (N, 2*dim).")
        if y.shape[1] != 2 * self.dim:
            raise ValueError(f"y0 debe tener {2 * self.dim} columnas.")

        N = len(y)
        eventos = tuple(eventos) 
        if lambda_max is None:
            lambda_max = self._lambda_max_auto(y, eventos)
        estado = np.empty_like(y0)
        status = np.full(N, STATUS_ACTIVE, dtype=np.int8)
        parametro = np.zeros(N) 
        pasos_aceptados = np.zeros(N, dtype=int)
        pasos_rechazados = np.zeros(N, dtype=int)

        y_active = y0.copy()
        h_active = np.full(N, h0, dtype=float)
        parametro_active = np.zeros(N, dtype=float)
        ids_active = np.arange(N, dtype=int)
        n_active = N

        self._ensure_buffers(N)
        b = self._buffers

        inicio = time.perf_counter()
        ultimo_reporte = inicio

        for _ in range(max_steps):
            if n_active == 0:
                break

            y_batch = y_active[:n_active]
            h_batch = h_active[:n_active]
            parametro_batch = parametro_active[:n_active]
            ids_batch = ids_active[:n_active]

            # reporte de avance 
            if progress:
                ahora = time.perf_counter()
                if ahora - ultimo_reporte >= progress_interval:
                    terminados = N - n_active
                    fraccion = terminados / N
                    elapsed = ahora - inicio
                    if terminados > 0:
                        eta = elapsed * (1.0 - fraccion) / fraccion
                        eta_txt = f"ETA~ {eta:7.1f} s"
                    else:
                        eta_txt = "ETA~    ---"

                    porcentaje = 100.0 * fraccion
                    ancho = 24
                    llenos = int(ancho * fraccion)
                    barra = "█" * llenos + "░" * (ancho - llenos)

                    print(
                        f"\rIntegrando [{barra}] "
                        f"{porcentaje:6.2f}% | "
                        f"finalizados={terminados:6d}/{N} | "
                        f"activos={n_active:6d} | "
                        f"t={elapsed:7.1f} s | "
                        f"{eta_txt}",
                        end="",
                        flush=True,
                    )

                    ultimo_reporte = ahora

            h_trial = b["h_trial"][:n_active]
            
            restante = lambda_max - parametro_batch
            np.minimum(h_batch, restante, out=h_trial)

            y_new, error = self._rk45(y_batch, h_trial)

            accepted = b["accepted"][:n_active]
            rejected = b["rejected"][:n_active]

            np.less_equal(error, 1.0, out=accepted)
            np.logical_not(accepted, out=rejected)

            finished = np.zeros(n_active, dtype=bool)

            if np.any(rejected):
                rej_idx = np.flatnonzero(rejected)
                factor = self._factor_paso(error[rej_idx], np.zeros(len(rej_idx), dtype=bool), out=b["factor"][:len(rej_idx)])
                h_nuevo = h_batch[rej_idx] * factor
                fallo = h_nuevo < h_min

                if np.any(fallo):
                    fallo_idx = rej_idx[fallo]
                    fallo_ids = ids_batch[fallo_idx]

                    status[fallo_ids] = STATUS_STEP_FAILURE
                    estado[fallo_ids] = y_batch[fallo_idx]
                    parametro[fallo_ids] = parametro_active[fallo_idx]
                    pasos_rechazados[fallo_ids] += 1
                    finished[fallo_idx] = True

                if np.any(~fallo):
                    valid_idx = rej_idx[~fallo]
                    h_batch[valid_idx] = h_nuevo[~fallo]
                    pasos_rechazados[ids_batch[valid_idx]] += 1

            if np.any(accepted):
                acc_idx = np.flatnonzero(accepted)
                ids_acc = ids_batch[acc_idx]

                y_prev = y_batch[acc_idx]
                y_next = y_new[acc_idx]
                h_acc = h_trial[acc_idx]
                error_acc = error[acc_idx]
                pasos_aceptados[ids_acc] += 1

                evento_alpha = np.full(len(acc_idx), np.inf, dtype=float)
                evento_code = np.full(len(acc_idx), -1, dtype=np.int8)

                for evento in eventos:
                    detectado, alpha = evento.detect(y_prev, y_next)
                    tomar = (detectado & (alpha < evento_alpha))

                    evento_alpha[tomar] = alpha[tomar]
                    evento_code[tomar] = evento.code

                ocurrio_evento = np.isfinite(evento_alpha)

                normales = ~ocurrio_evento
                if np.any(normales):
                    normal_idx = acc_idx[normales]
                    y_batch[normal_idx] = y_next[normales] 
                    parametro_active[normal_idx] += (h_acc[normales])
                    factor = self._factor_paso(error_acc[normales], np.ones(np.sum(normales), dtype=bool), out=b["factor"][:np.sum(normales)])
                    h_active[normal_idx] *= factor
                    h_active[normal_idx] = np.clip(h_active[normal_idx], h_min, h_max) 

                if np.any(ocurrio_evento):
                    event_local = acc_idx[ocurrio_evento]
                    event_ids = ids_acc[ocurrio_evento]
                    alpha_event = (evento_alpha[ocurrio_evento])
                    y_hit = (y_prev[ocurrio_evento] + alpha_event[:, None] * (y_next[ocurrio_evento] - y_prev[ocurrio_evento]))
                    parametro_event = (parametro_active[event_local] + alpha_event * h_acc[ocurrio_evento])
                    estado[event_ids] = y_hit
                    parametro[event_ids] = parametro_event
                    status[event_ids] = evento_code[ocurrio_evento]
                    finished[event_local] = True

                if np.any(normales):
                    normal_idx = acc_idx[normales]
                    llego = (parametro_active[normal_idx] >= lambda_max - EPS)
                    if np.any(llego):
                        lambda_idx = normal_idx[llego]
                        lambda_ids = (ids_batch[lambda_idx])
                        estado[lambda_ids] = y_batch[lambda_idx]
                        parametro[lambda_ids] = parametro_active[lambda_idx]
                        status[lambda_ids] = STATUS_MAX_LAMBDA
                        finished[lambda_idx] = True

            if np.any(finished):
                keep_idx = np.flatnonzero(~finished)
                new_n = len(keep_idx)

                y_active[:new_n] = y_active[keep_idx]
                h_active[:new_n] = h_active[keep_idx]
                parametro_active[:new_n] = parametro_active[keep_idx]
                ids_active[:new_n] = ids_active[keep_idx]
                n_active = new_n

        if n_active > 0:
            ids_remaining = ids_active[:n_active]
            estado[ids_remaining] = y_active[:n_active]
            parametro[ids_remaining] = parametro_active[:n_active]

        if progress:
            elapsed = time.perf_counter() - inicio
            if n_active == 0:
                print(
                    f"\rIntegrando ["
                    f"{'█' * 24}"
                    f"] 100.00% | "
                    f"finalizados={N:6d}/{N} | "
                    f"activos={0:6d} | "
                    f"t={elapsed:7.1f} s | "
                    f"ETA~    0.0 s"
                    + " " * 10
                )
            else:
                terminados = N - n_active
                porcentaje = (100.0 * terminados / N)
                print(
                    f"\rIntegración detenida | "
                    f"{porcentaje:6.2f}% | "
                    f"finalizados={terminados:6d}/{N} | "
                    f"activos={n_active:6d} | "
                    f"t={elapsed:7.1f} s"
                    + " " * 10
                )

        return RayTracingResult(
            estado=estado,
            status=status,
            parametro=parametro,
            pasos_aceptados=pasos_aceptados,
            pasos_rechazados=pasos_rechazados,
        )   

