from pathlib import Path
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from Metricas.schwarzschild import Schwarzschild
from Solver.ray_tracing import (
    SolverRayTracing,
    EscapeEvent,
    HorizonEvent,
    STATUS_ACTIVE,
    STATUS_ESCAPE,
    STATUS_HORIZON,
    STATUS_DISK,
    STATUS_MAX_LAMBDA,
    STATUS_STEP_FAILURE,
)
from Tensores.operaciones import tetrada_obs, transformar_vector
from Visualizador.RayTracing.cam import Camara


# -----------------------------------------------------------------------------
# Configuracion del benchmark
# -----------------------------------------------------------------------------
M = 1.0
R0 = 100.0
FOV = 60.0

# La prueba es deliberadamente menor que la imagen final. Una vez elegido el
# esquema, estos valores se pueden repetir con 250x250.
RESOLUCION = (64, 64)

# Para diagnostico fisico conviene detectar el retorno a la esfera del
# observador. Un rayo que entra inicialmente solo puede cruzarla hacia afuera
# despues de girar; esto evita depender de R_MAX > R0 durante el benchmark.
R_ESCAPE = R0
R_STOP = 2.001
LAMBDA_MAX = 400.0

# La prueba de rendimiento barre h_max. La tolerancia se puede endurecer luego.
RTOL = 1e-7
ATOL = 1e-9
H0 = 0.01
H_MIN = 1e-10
H_MAX_VALUES = (0.05, 0.1, 0.2, 0.5, 1.0, np.inf)


STATUS_NAMES = {
    STATUS_ACTIVE: "active",
    STATUS_ESCAPE: "escape",
    STATUS_HORIZON: "horizon",
    STATUS_DISK: "disk",
    STATUS_MAX_LAMBDA: "max_lambda",
    STATUS_STEP_FAILURE: "step_failure",
}


class InstrumentedSolver(SolverRayTracing):
    """Mide el trabajo batch real sin cambiar el algoritmo del solver."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_metrics()

    def reset_metrics(self):
        self.rhs_calls = 0
        self.rhs_points = 0
        self.h_samples = []

    def rhs(self, y):
        self.rhs_calls += 1
        self.rhs_points += len(y)
        return super().rhs(y)

    def _rk45(self, y, h):
        self.h_samples.append(np.asarray(h, dtype=float).copy())
        return super()._rk45(y, h)


def construir_observador(metrica):
    x0 = np.array([0.0, R0, np.pi / 2.0, 0.0])

    metric_num = metrica.numeric("metric")
    valores = metric_num.evaluar_valores(*x0)
    g = np.zeros((4, 4), dtype=float)
    for idx, valor in zip(metric_num.indices, valores):
        g[idx] = valor

    u_obs = np.zeros(4)
    u_obs[0] = 1.0 / np.sqrt(-g[0, 0])
    tetrada = tetrada_obs(metrica, x0, u_obs)
    return x0, tetrada


def construir_rayos(metrica, x0, tetrada):
    camara = Camara(
        posicion=x0,
        resolucion=RESOLUCION,
        fov=np.radians(FOV),
        foward=(-1.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
    )
    k_local = camara.rays_local()
    k = transformar_vector(tetrada, k_local)

    y0 = np.empty((len(k), 8), dtype=float)
    y0[:, :4] = x0
    y0[:, 4:] = k
    return camara, y0


def diagnostico_inicial(metrica, y0):
    """Comprueba que la camara genera la distribucion de impacto esperada."""
    x = y0[:, :4]
    u = y0[:, 4:]

    metric_num = metrica.numeric("metric")
    valores = metric_num.evaluar_valores(*x.T)
    # Schwarzschild es diagonal; en este punto solo necesitamos g_tt y g_phiphi.
    # evaluar_valores puede devolver arrays para los puntos de entrada.
    mapa = {idx: np.asarray(v) for idx, v in zip(metric_num.indices, valores)}

    gtt = mapa[(0, 0)]
    gphiphi = mapa[(3, 3)]

    energy = -gtt * u[:, 0]
    angular = gphiphi * u[:, 3]
    impact = np.abs(angular / energy)

    bc = 3.0 * np.sqrt(3.0) * M
    falling = impact < bc

    print()
    print("DIAGNOSTICO DE LA CAMARA")
    print("-" * 80)
    print(f"b_critico          = {bc:.6f}")
    print(f"b min / max        = {impact.min():.6f} / {impact.max():.6f}")
    print(f"b mediana           = {np.median(impact):.6f}")
    print(f"rayos b < b_critico = {falling.sum()} / {len(impact)} ({100.0 * falling.mean():.2f} %)")

    if np.all(impact < bc):
        print("ADVERTENCIA: toda la camara cae dentro del cono critico.")
    elif np.all(impact > bc):
        print("ADVERTENCIA: ningun rayo entra en el cono critico.")
    else:
        print("OK: la camara contiene rayos que caen y rayos que escapan.")


def imprimir_estados(status):
    print()
    print("ESTADOS")
    print("-" * 80)
    for code in range(6):
        n = np.count_nonzero(status == code)
        print(f"{STATUS_NAMES[code]:12s}= {n}")


def imprimir_trabajo(solver):
    h = np.concatenate(solver.h_samples) if solver.h_samples else np.empty(0)
    print()
    print("TRABAJO REAL DEL BATCH")
    print("-" * 80)
    print(f"RHS calls          = {solver.rhs_calls}")
    print(f"puntos procesados  = {solver.rhs_points:,}")

    if h.size == 0:
        return

    q = np.percentile(h, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    print("h usado en RK45")
    print(f"min/1%/5%          = {q[0]:.3e} / {q[1]:.3e} / {q[2]:.3e}")
    print(f"25/50/75%          = {q[3]:.3e} / {q[4]:.3e} / {q[5]:.3e}")
    print(f"95/99%/max         = {q[6]:.3e} / {q[7]:.3e} / {q[8]:.3e}")


def run_case(metrica, y0, h_max, *, lambda_max=LAMBDA_MAX):
    solver = InstrumentedSolver(metrica, rtol=RTOL, atol=ATOL)
    eventos = (
        HorizonEvent(R_STOP, radial_index=1),
        EscapeEvent(R_ESCAPE, radial_index=1),
    )

    inicio = time.perf_counter()
    resultado = solver.resolver(
        y0,
        lambda_max=lambda_max,
        h0=H0,
        h_min=H_MIN,
        h_max=h_max,
        eventos=eventos,
    )
    elapsed = time.perf_counter() - inicio

    steps = resultado.pasos_aceptados
    rejected = resultado.pasos_rechazados

    return resultado, solver, elapsed, steps, rejected


def imprimir_escala_por_rayo(resultado, steps, rejected):
    active_or_terminal = np.ones(len(steps), dtype=bool)
    del active_or_terminal

    print()
    print("PASOS POR RAYO")
    print("-" * 80)
    q_steps = np.percentile(steps, [0, 25, 50, 75, 95, 99, 100])
    q_rej = np.percentile(rejected, [0, 50, 95, 99, 100])
    print(
        "aceptados  min/25/50/75/95/99/max = "
        + " / ".join(f"{x:.0f}" for x in q_steps)
    )
    print(
        "rechazados min/50/95/99/max          = "
        + " / ".join(f"{x:.0f}" for x in q_rej)
    )
    print(f"aceptados totales                   = {steps.sum():,}")
    print(f"rechazados totales                  = {rejected.sum():,}")


def benchmark_hmax(metrica, y0):
    print()
    print("BENCHMARK h_max")
    print("=" * 80)
    print(
        f"resolucion={RESOLUCION}, FOV={FOV} deg, r0={R0}, "
        f"lambda_max={LAMBDA_MAX}, r_escape={R_ESCAPE}"
    )
    print(
        f"rtol={RTOL:.1e}, atol={ATOL:.1e}, h0={H0}, h_min={H_MIN}"
    )
    print()
    print(
        f"{'h_max':>10s} {'tiempo[s]':>12s} {'escape':>10s} "
        f"{'horizon':>10s} {'maxlam':>10s} {'pasos':>14s} {'rechazos':>14s}"
    )
    print("-" * 94)

    resultados = []
    for h_max in H_MAX_VALUES:
        resultado, solver, elapsed, steps, rejected = run_case(metrica, y0, h_max)
        counts = [np.count_nonzero(resultado.status == c) for c in range(6)]
        print(
            f"{str(h_max):>10s} {elapsed:12.3f} {counts[STATUS_ESCAPE]:10d} "
            f"{counts[STATUS_HORIZON]:10d} {counts[STATUS_MAX_LAMBDA]:10d} "
            f"{steps.sum():14,d} {rejected.sum():14,d}"
        )
        resultados.append((h_max, resultado, solver, elapsed, steps, rejected))

    return resultados


def main():
    global RESOLUCION, FOV, R0

    # Permite ampliar la prueba sin editar el archivo.
    size = int(os.environ.get("RT_TEST_SIZE", RESOLUCION[0]))
    RESOLUCION = (size, size)
    FOV = float(os.environ.get("RT_TEST_FOV", FOV))
    R0 = float(os.environ.get("RT_TEST_R0", R0))

    metrica = Schwarzschild(M=M)
    x0, tetrada = construir_observador(metrica)
    _, y0 = construir_rayos(metrica, x0, tetrada)

    print("=" * 80)
    print("SCHWARZSCHILD — RAY TRACING DIAGNOSTIC / PERFORMANCE TEST")
    print("=" * 80)
    print(f"resolution = {RESOLUCION}")
    print(f"FOV        = {FOV} deg")
    print(f"r0         = {R0}")
    print(f"r_stop     = {R_STOP}")
    print(f"r_escape   = {R_ESCAPE}")
    print(f"lambda_max = {LAMBDA_MAX}")

    diagnostico_inicial(metrica, y0)

    # Una corrida de referencia con h_max=0.1 permite inspeccionar todos los
    # estados y medir la distribucion de h antes del sweep.
    print()
    print("CORRIDA DE REFERENCIA")
    print("=" * 80)
    resultado, solver, elapsed, steps, rejected = run_case(metrica, y0, 0.1)
    print(f"tiempo = {elapsed:.3f} s")
    imprimir_estados(resultado.status)
    imprimir_trabajo(solver)
    imprimir_escala_por_rayo(resultado, steps, rejected)

    # Esta prueba es la que decide si el agrupamiento por h merece trabajo
    # adicional: no implementa un solver alternativo; mide el trabajo real que
    # el algoritmo actual ya realiza. La cantidad de RHS ponderada por cantidad
    # de rayos es el coste vectorizado que habria que reempaquetar al agrupar.
    total_steps = int(steps.sum())
    print()
    print("DECISION SOBRE AGRUPAMIENTO POR h")
    print("-" * 80)
    print(f"pasos por rayo totales               = {total_steps:,}")
    print(f"evaluaciones RHS teoricas (x7)       = {7 * total_steps:,}")
    print(
        "El agrupamiento solo puede mejorar si reduce costes de memoria/allocation "
        "o permite kernels con paso fijo; no reduce automaticamente estas etapas RK45."
    )

    resultados = benchmark_hmax(metrica, y0)

    # Seleccion simple: entre configuraciones con la misma fisica, tomar la mas
    # rapida que no cambie el conjunto de escapes/horizontes de la referencia.
    ref_counts = np.bincount(resultado.status, minlength=6)
    candidates = []
    for h_max, result_i, solver_i, time_i, steps_i, rejected_i in resultados:
        counts_i = np.bincount(result_i.status, minlength=6)
        same_physics = (
            counts_i[STATUS_ESCAPE] == ref_counts[STATUS_ESCAPE]
            and counts_i[STATUS_HORIZON] == ref_counts[STATUS_HORIZON]
        )
        if same_physics:
            candidates.append((time_i, h_max))

    print()
    print("CONCLUSION AUTOMATICA DEL SWEEP")
    print("-" * 80)
    if candidates:
        best_time, best_h = min(candidates)
        print(f"mejor h_max dentro del sweep = {best_h}")
        print(f"tiempo                         = {best_time:.3f} s")
    else:
        print("Ningun h_max del sweep reprodujo exactamente los conteos de la referencia.")
        print("Eso indica que hay que controlar primero la fisica/tolerancia antes de optimizar.")


if __name__ == "__main__":
    main()
