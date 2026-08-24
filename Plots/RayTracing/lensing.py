from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from Metricas.schwarzschild import Schwarzschild
from Solver.ray_tracing import (SolverRayTracing, EscapeEvent, HorizonEvent)
from Tensores.operaciones import (tetrada_obs, transformar_vector)
from Visualizador.RayTracing.cam import Camara
from Visualizador.RayTracing.background import GridBackground
from Visualizador.RayTracing.render import RayTracingRenderer

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
R_MAX = 100.0    # > R0

BACKGROUND_DISTANCE = 100.0
BACKGROUND_EXTENT = 60.0
GRID_SPACING = 10.0

def construir_observador(metrica):
    x0 = np.array([0.0, R0, np.pi / 2.0, 0.0])
    metric_num = metrica.numeric("metric")
    valores = (metric_num.evaluar_valores(*x0))
    g = np.zeros((4, 4), dtype=float)

    for idx, valor in zip(metric_num.indices, valores):
        g[idx] = valor

    u_obs = np.zeros(4)
    u_obs[0] = (1.0 / np.sqrt(-g[0, 0]))
    e = tetrada_obs(metrica, x0, u_obs)

    return x0, e

def construir_rayos(metrica, x0, tetrada):
    camara = Camara(posicion=x0, resolucion=RESOLUCION, fov=np.radians(FOV), foward=(-1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0))
    k_local = camara.rays_local()
    k = transformar_vector(tetrada, k_local)
    y0 = np.empty((len(k), 8), dtype=float)
    y0[:, :4] = x0
    y0[:, 4:] = k
    return camara, y0

def main(save=False, show=True):
    print("=" * 80)
    print("SCHWARZSCHILD — GRAVITATIONAL LENSING")
    print("=" * 80)

    metrica = Schwarzschild(M=M)
    x0, tetrada = (construir_observador(metrica))
    camara, y0 = construir_rayos(metrica, x0, tetrada)

    print()
    print(f"resolución = {RESOLUCION}")
    print(f"FOV        = {FOV}°")
    print(f"rayos      = {len(y0)}")

    solver = SolverRayTracing(metrica, rtol=RTOL, atol=ATOL)
    eventos = (HorizonEvent(R_STOP, radial_index=1), EscapeEvent(R_MAX, radial_index=1))

    print()
    print("Integrando...")

    inicio = time.perf_counter()

    resultado = solver.resolver(y0, lambda_max=LAMBDA_MAX, h0=H0, h_min=H_MIN, h_max=H_MAX, eventos=eventos, progress=True, progress_interval=10.0)

    tiempo = (time.perf_counter() - inicio)

    print(f"tiempo = {tiempo:.3f} s")

    for code, nombre in [
        (0, "active"),
        (1, "escape"),
        (2, "horizon"),
        (3, "disk"),
        (4, "max_lambda"),
        (5, "step_failure"),
    ]:
        print(f"{nombre:12s}= {np.count_nonzero(resultado.status == code)}")

    background = GridBackground(distance=BACKGROUND_DISTANCE, extent=BACKGROUND_EXTENT, spacing=GRID_SPACING, subgrid=True)
    renderer = RayTracingRenderer(background)
    image = renderer.render(resultado, camara, background_distance=BACKGROUND_DISTANCE)


    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, origin="upper", interpolation="nearest")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if save:
        output = (ROOT / "Plots" / "graficos" / "lensing_sch_grid.png")
        fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0)
        print()
        print(F"Guardado: {output}")

    if show:
        plt.show()

    return image, resultado

if __name__ == "__main__":
    main()