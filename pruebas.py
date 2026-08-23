import time
import numpy as np

from Metricas.schwarzschild import Schwarzschild
from Solver.ray_tracing import SolverRayTracing
from Tensores.operaciones import tetrada_obs, transformar_vector
from Visualizador.RayTracing.cam import Camara


# ============================================================
# CONFIGURACIÓN
# ============================================================

metrica = Schwarzschild(M=1.0)

x0 = np.array([
    0.0,
    10.0,
    np.pi / 2,
    0.0,
])

FOVS = (
    20.0,
    60.0,
    120.0,
)

TOLERANCIAS = (
    (1e-6, 1e-8),
    (1e-9, 1e-11),
)


# ============================================================
# OBSERVADOR
# ============================================================

metric_num = metrica.numeric("metric")
valores = metric_num.evaluar_valores(*x0)

g0 = np.zeros((4, 4), dtype=float)

for idx, valor in zip(metric_num.indices, valores):
    g0[idx] = valor

u_obs = np.zeros(4)
u_obs[0] = 1.0 / np.sqrt(-g0[0, 0])

e = tetrada_obs(
    metrica,
    x0,
    u_obs,
)


# ============================================================
# BENCHMARK
# ============================================================

print("=" * 80)
print("BENCHMARK FOV vs TOLERANCIA")
print("=" * 80)

print(
    f"{'FOV':>8} "
    f"{'rtol':>10} "
    f"{'tiempo [s]':>14} "
    f"{'us/rayo':>14} "
    f"{'pasos prom.':>14} "
    f"{'pasos min':>12} "
    f"{'pasos max':>12} "
    f"{'rechazos':>12}"
)

print("-" * 80)


for fov_deg in FOVS:

    camara = Camara(
        posicion=x0,
        resolucion=(100, 100),
        fov=np.radians(fov_deg),
    )

    k_local = camara.rays_local()

    k = transformar_vector(
        e,
        k_local,
    )

    N = len(k)

    y0 = np.empty(
        (N, 8),
        dtype=float,
    )

    y0[:, :4] = x0
    y0[:, 4:] = k

    for rtol, atol in TOLERANCIAS:

        solver = SolverRayTracing(
            metrica,
            rtol=rtol,
            atol=atol,
        )

        t0 = time.perf_counter()

        resultado = solver.resolver(
            y0,
            lambda_max=20.0,
            h0=0.01,
            h_min=1e-10,
            h_max=0.1,
        )

        tiempo = time.perf_counter() - t0

        pasos = resultado.pasos_aceptados

        print(
            f"{fov_deg:8.0f} "
            f"{rtol:10.0e} "
            f"{tiempo:14.6f} "
            f"{tiempo / N * 1e6:14.3f} "
            f"{pasos.mean():14.2f} "
            f"{pasos.min():12d} "
            f"{pasos.max():12d} "
            f"{resultado.pasos_rechazados.sum():12d}"
        )


print("\n" + "=" * 80)
print("FIN")
print("=" * 80)