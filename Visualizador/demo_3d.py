"""Ejemplo de visualización 3D del embedding de Schwarzschild.

Ejecutar desde la raíz del repositorio:
    python -m Visualizador.demo_3d
"""

import numpy as np
import matplotlib.pyplot as plt

from Metricas.schwarzschild import Schwarzschild
from Solver.solver import SolverGeodesica
from Solver.invariantes import Condiciones
from Visualizador.embedding import Embedding
from Visualizador.escena import Visualizador3D


def construir_geodesica(solver, cond, r0, vr, vphi, nombre):
    """Construye una geodésica timelike con normalización automática."""
    x0 = np.array([0.0, r0, np.pi / 2, 0.0])
    u0 = cond.normalizar(
        x0,
        np.array([1.0, vr, 0.0, vphi]),
        tipo="timelike",
        componente=0,
        signo=+1,
    )
    resultado = solver.resolver(
        x0,
        u0,
        (0.0, 100.0),
        metodo="DOP853",
        rtol=1e-9,
        atol=1e-11,
    )
    return resultado, nombre


def main():
    metric = Schwarzschild(M=1.0)
    solver = SolverGeodesica(metric)
    cond = Condiciones(metric)

    embedding = Embedding(
        metric,
        coordenadas=(1, 3),
        fijas={0: 0.0, 2: np.pi / 2},
    )

    escena = Visualizador3D(
        embedding,
        qmin=2.001,
        qmax=25.0,
        nq=350,
        nphi=180,
        cmap="viridis",
    )

    condiciones = [
        (10.0, -0.08, 0.045, "Caída 1"),
        (12.0, -0.02, 0.075, "Caída 2"),
        (16.0,  0.03, 0.060, "Órbita / escape"),
        (20.0,  0.06, 0.035, "Trayectoria 4"),
    ]

    for r0, vr, vphi, nombre in condiciones:
        resultado, nombre = construir_geodesica(
            solver, cond, r0, vr, vphi, nombre
        )
        if resultado.success:
            escena.add_geodesica(resultado, nombre=nombre)

    fig, ax = escena.draw(
        title="Geodésicas en el embedding de Schwarzschild",
        elev=28,
        azim=-55,
        surface_alpha=0.68,
        show_legend=True,
        show=False,
    )

    fig.savefig(
        "embedding_schwarzschild.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
