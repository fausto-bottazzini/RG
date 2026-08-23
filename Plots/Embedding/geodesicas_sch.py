from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from Metricas.schwarzschild import Schwarzschild
from Solver.solver import SolverGeodesica
from Solver.invariantes import Condiciones

from Visualizador.embedding import Embedding
from Visualizador.escena import Visualizador2D
from Plots.Embedding._common import construir_geodesica

def main():
    M = 1.0
    metrica = Schwarzschild(M=M)

    print("=" * 70)
    print("VISUALIZACIÓN 3D — SCHWARZSCHILD")
    print(f"M = {M}")

    solver = SolverGeodesica(metrica)
    condiciones = Condiciones(metrica)
    embedding = Embedding(metrica, coordenadas=(1, 3), fijas={0: 0.0, 2: np.pi / 2})
    escena = Visualizador2D(embedding, qmin=2.001, qmax=100.0, nq=80, nphi=30, figsize=(9, 16), cmap="gist_heat_r")
    configuraciones = [
        (10.0, 0.0, 0.037, "Órbita Circular"),
        (30.0, -0.025, 0.0045, "Preseción"), # 30.0, -0.023, 0.0042
        (50.0, 0.0, 0.0, "Caida"),
        (70.0,  0.0, 0.000799, "Trayectoria exterior"),
    ]

    for r0, vr, vphi, nombre in configuraciones:
        resultado = construir_geodesica(solver, condiciones, r0, vr, vphi, tipo="timelike", lambda_max=5000)
        if resultado.y.shape[1] >= 2:
            escena.add_geodesica(resultado, nombre=nombre, linewidth=1.5, particle_size=7)

    fig, ax = escena.ensure_axes()
    ax.set_axis_off()
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # fig, ax = escena.draw(
    #     title="Geodésicas en el embedding de Schwarzschild",
    #     surface_alpha=0.8,
    #     elev=30,
    #     azim=-55,
    #     show_legend=True,
    #     show=True,
    # )

    # ----------------------------------------------------------
    # Animación
    # ----------------------------------------------------------

    escena.animate(
        frames=600,
        interval=30,
        trail_length=50,
        title="Geodésicas sobre el embedding de Schwarzschild",
        elev=30,
        azim=-130,
        surface_alpha=0.8,
        show_legend=False,
        repeat=True,
        show=True,
    )

    # escena.save_animation(
    #     "Plots/graficos/geodesica_schwarzschild.mp4",
    #     fps=30,
    #     dpi=100,
    #     bitrate=8000,
    #     frames=600,
    #     interval=30,
    #     trail_length=100,
    #     title="Geodésicas sobre el embedding de Schwarzschild",
    #     elev=30,
    #     azim=-130,
    #     surface_alpha=0.8,
    #     show_legend=False,
    # )

if __name__ == "__main__":
    main()