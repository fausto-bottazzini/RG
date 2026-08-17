from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from Metricas.schwarzschild import Schwarzschild
from Solver.solver import SolverGeodesica
from Solver.invariantes import Condiciones

from Visualizador.embedding import Embedding
from Visualizador.escena import Visualizador3D
from Plots.Embedding._common import construir_geodesica

def main(ani=False, save=False):
    M = 1.0
    a = 0.8
    metrica = Schwarzschild(M=M)

    print("=" * 70)
    print("VISUALIZACIÓN 3D — SCHWARZSCHILD")
    print(f"M = {M}")

    solver = SolverGeodesica(metrica)
    condiciones = Condiciones(metrica)
    embedding = Embedding(metrica, coordenadas=(1, 3), fijas={0: 0.0, 2: np.pi / 2})
    escena = Visualizador3D(embedding, qmin=2.001, qmax=200.0, nq=80, nphi=36, figsize=(9, 16), cmap="inferno_r")
    configuraciones = [
        (170.0, -0.1, 0.00005, "Ray 1"),
        (170.0, -0.1, 0.00008, "Ray 2"), 
        (170.0, -0.1, -0.00005, "Ray -1"),
        (170.0,  -0.1, -0.00008, "Ray -2"),
    ]

    for r0, vr, vphi, nombre in configuraciones:
        resultado = construir_geodesica(solver, condiciones, r0, vr, vphi, tipo="null", lambda_max=5100.0)
        if resultado.y.shape[1] >= 2:
            escena.add_geodesica(resultado, nombre=nombre, linewidth=1, particle_size=7, color="khaki")

    if not ani:
        fig, ax = escena.ensure_axes()
        ax.set_axis_off()
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        fig, ax = escena.draw(
            title="Geodésicas en el embedding de Schwarzschild",
            surface_alpha=0.8,
            elev=30,
            azim=-55,
            show_legend=False,
            show=True,
        )


    # ----------------------------------------------------------
    # Animación
    # ----------------------------------------------------------

    fig, ax = escena.ensure_axes()
    ax.set_axis_off()
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    elev = 34
    azim = -160

    if ani and not save:

        escena.animate(
            frames=600,
            interval=30,
            trail_length=500,
            title="Geodésicas nulas sobre el embedding de Schwarzschild",
            elev=elev,
            azim=azim,
            surface_alpha=0.8,
            show_legend=False,
            repeat=True,
            show=True,
        )

    if ani and save:
            escena.save_animation(
                "Plots/graficos/lensing_sch.mp4",
                fps=30,
                dpi=100,
                bitrate=8000,
                frames=600,
                interval=30,
                trail_length=500,
                title="Geodésicas nulas sobre el embedding de Schwarzschild",
                elev=elev,
                azim=azim,
                surface_alpha=0.8,
                show_legend=False,
            )

    
if __name__ == "__main__":
    main(ani=False, save=False)