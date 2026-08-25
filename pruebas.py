from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from Metricas.schwarzschild import Schwarzschild
from Solver.ray_tracing import SolverRayTracing, EscapeEvent, HorizonEvent
from Tensores.operaciones import tetrada_obs, transformar_vector
from Visualizador.RayTracing.cam import Camara


M = 1.0
R0 = 50.0

RESOLUCION = (250, 250)
FOV = 60.0

RTOL = 1e-9
ATOL = 1e-11

H0 = 0.01
H_MIN = 1e-10
H_MAX = np.inf

LAMBDA_MAX = 1000.0

R_STOP = 2.001
R_MAX = 100.0

BATCH_VALUES = (
    3930,
    3940,
    3860,
    3970,
    3980,
)


def construir_observador(metrica):
    x0 = np.array([0.0, R0, np.pi / 2.0, 0.0])

    metric_num = metrica.numeric("metric")
    valores = metric_num.evaluar_valores(*x0)

    g = np.zeros((4, 4), dtype=float)

    for idx, valor in zip(metric_num.indices, valores):
        g[idx] = valor

    u_obs = np.zeros(4)
    u_obs[0] = 1.0 / np.sqrt(-g[0, 0])

    e = tetrada_obs(
        metrica,
        x0,
        u_obs,
    )

    return x0, e


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


def ejecutar(metrica, y0, batch_max):
    solver = SolverRayTracing(
        metrica,
        rtol=RTOL,
        atol=ATOL,
        batch_max=batch_max,
    )

    eventos = (
        HorizonEvent(
            R_STOP,
            radial_index=1,
        ),
        EscapeEvent(
            R_MAX,
            radial_index=1,
        ),
    )

    inicio = time.perf_counter()

    resultado = solver.resolver(
        y0,
        lambda_max=LAMBDA_MAX,
        h0=H0,
        h_min=H_MIN,
        h_max=H_MAX,
        eventos=eventos,
        progress=False,
    )

    elapsed = time.perf_counter() - inicio

    return resultado, elapsed


def main():
    print("=" * 90)
    print("SCHWARZSCHILD — SWEEP batch_max / CONFIGURACIÓN REAL DE LENSING")
    print("=" * 90)

    print(f"resolución = {RESOLUCION}")
    print(f"FOV        = {FOV}°")
    print(f"R0         = {R0}")
    print(f"R_MAX      = {R_MAX}")
    print(f"rayos      = {RESOLUCION[0] * RESOLUCION[1]:,}")
    print(f"rtol/atol  = {RTOL:.1e} / {ATOL:.1e}")
    print(f"h0         = {H0}")
    print(f"h_min      = {H_MIN}")
    print(f"h_max      = {H_MAX}")
    print(f"lambda_max = {LAMBDA_MAX}")

    metrica = Schwarzschild(M=M)
    _, y0 = construir_rayos(
        metrica,
        *construir_observador(metrica),
    )

    print()
    print(
        f"{'batch_max':>12s} "
        f"{'tiempo [s]':>14s} "
        f"{'speedup':>10s} "
        f"{'escape':>10s} "
        f"{'horizon':>10s} "
        f"{'max_lambda':>12s} "
        f"{'failure':>10s}"
    )
    print("-" * 90)

    resultados = []

    for batch_max in BATCH_VALUES:
        resultado, elapsed = ejecutar(
            metrica,
            y0,
            batch_max,
        )

        escape = np.count_nonzero(
            resultado.status == 1
        )

        horizon = np.count_nonzero(
            resultado.status == 2
        )

        max_lambda = np.count_nonzero(
            resultado.status == 4
        )

        failure = np.count_nonzero(
            resultado.status == 5
        )

        resultados.append(
            (
                batch_max,
                elapsed,
                resultado,
            )
        )

        print(
            f"{batch_max:12d} "
            f"{elapsed:14.3f} "
            f"{1.0:10.3f} "
            f"{escape:10d} "
            f"{horizon:10d} "
            f"{max_lambda:12d} "
            f"{failure:10d}"
        )

    mejor = min(
        resultados,
        key=lambda x: x[1],
    )

    print()
    print("=" * 90)
    print("RESULTADO")
    print("=" * 90)
    print(f"mejor batch_max = {mejor[0]}")
    print(f"tiempo          = {mejor[1]:.3f} s")

    print()
    print("Comparación física:")

    for batch_max, elapsed, resultado in resultados:
        print(
            f"{batch_max:6d}: "
            f"escape={np.count_nonzero(resultado.status == 1):6d}, "
            f"horizon={np.count_nonzero(resultado.status == 2):6d}, "
            f"max_lambda={np.count_nonzero(resultado.status == 4):6d}, "
            f"failure={np.count_nonzero(resultado.status == 5):6d}"
        )


if __name__ == "__main__":
    main()