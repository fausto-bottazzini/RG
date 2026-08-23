from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from Metricas.kerr import Kerr
from Solver.solver import SolverGeodesica
from Solver.invariantes import Condiciones
from Visualizador.embedding import Embedding
from Visualizador.escena import Visualizador2D
from matplotlib.animation import FuncAnimation

M = 1.0
a = 0.99

R_MIN = 2.05
R_MAX = 20.0

R0 = 20.0
VR = 0.0
VPHI = -0.01

steps = 500
NQ = 80
NPHI = 30

def construir_geodesica(metric, r0, vr, vphi):
    solver = SolverGeodesica(metric)
    cond = Condiciones(metric)
    x0 = np.array([0.0, r0, np.pi / 2, 0.0])
    u0 = cond.normalizar(x0, np.array([1.0, vr, 0.0, vphi]), tipo="timelike", componente=0, signo=+1)
    resultado = solver.resolver(x0, u0, (0.0, steps), metodo="DOP853", rtol=1e-9, atol=1e-11)
    return resultado

def omega_frame_dragging(metric, r):
    """Velocidad angular de los ZAMO."""
    theta = np.pi / 2
    Sigma = r**2 + a**2 * np.cos(theta)**2
    g_tphi = (-2.0 * M * r * a * np.sin(theta)**2 / Sigma)
    g_phiphi = ((r**2 + a**2 + 2.0 * M * r * a**2 * np.sin(theta)**2 / Sigma) * np.sin(theta)**2)
    return -g_tphi / g_phiphi

def agregar_frame_dragging(embedding, rmin, rmax, nr=12, nphi=18):
    """Dibuja el campo de arrastre sobre la superficie.
    Las flechas son tangentes a las órbitas azimutales.
    Su módulo representa omega(r).
    """
    r_values = np.linspace(rmin, rmax, nr)
    phi_values = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    rho_values, z_values = embedding.profile(r_values)

    omega_values = omega_frame_dragging( embedding.metrica, r_values)
    omega_max = np.max(np.abs(omega_values))

    if omega_max == 0:
        return

    campo = []

    for r, rho, z, omega in zip(r_values, rho_values, z_values, omega_values):
        magnitude = np.abs(omega) / omega_max
        magnitude = np.sqrt(magnitude)
        direction = np.sign(omega)

        for phi in phi_values:
            campo.append({"rho": rho, "z": z, "phi": phi, "magnitude": magnitude, "direction": direction})

    return campo

def dibujar_frame_dragging(ax, campo, fase=0.0, arrow_scale=1.5):
    """Dibuja el campo de arrastre para una fase angular dada."""
    artists = []

    for dato in campo:
        rho = dato["rho"]
        z = dato["z"]
        phi = dato["phi"] + fase

        magnitude = dato["magnitude"]
        direction = dato["direction"]

        x = rho * np.cos(phi)
        y = rho * np.sin(phi)

        u = (-np.sin(phi) * magnitude * direction * arrow_scale)
        v = (np.cos(phi) * magnitude * direction * arrow_scale)
        q = ax.quiver(x, y, z, u, v, 0.0, length=1.0, normalize=False, color="white", linewidth=0.7, arrow_length_ratio=0.28, alpha=0.65)

        artists.append(q)

    return artists

def animar_kerr(escena, campo, *, frames=600, interval=30, trail_length=500, elev=30, azim=-55):
    """Animación completa."""

    if not escena.geodesicas:
        raise ValueError("No hay geodesicas para animar.")
    
    X, Y, Z = escena.build_surface()
    fig, ax = escena.ensure_axes()
    ax.plot_surface(X, Y, Z, cmap=escena.cmap, alpha=0.38, linewidth=0, antialiased=True, rcount=min(escena.nq, 220), ccount=min(escena.nphi, 140))

    dibujar_frame_dragging(ax, campo, fase=0.0, arrow_scale=1.5)

    geo = escena.geodesicas[0]
    t = np.asarray(getattr(geo.resultado, "t", np.arange(len(geo.x))), dtype=float)

    if len(t) != len(geo.x):
        t = np.linspace(0.0, 1.0, len(geo.x))

    timeline = np.linspace(t[0], t[-1], frames)

    color = geo.color if geo.color is not None else "white"
    line, = ax.plot([],[],[], color=geo.color, linewidth=geo.linewidth)
    particle, = ax.plot([],[],[], marker="o", linestyle="", color=color, markeredgecolor="white", markeredgewidth=0.8, markersize=max(4.0, np.sqrt(geo.particle_size)))

    def update(frame):
        current = timeline[frame]

        if current <= t[0]:
            j = 1
            tq = current
        elif current >= t[-1]:
            j = len(t) - 1
            tq = t[-1]
        else:
            j = int(np.searchsorted(t, current, side="right"))
            tq = current

        x_now = np.interp(tq, t[:j + 1], geo.x[:j + 1])
        y_now = np.interp(tq, t[:j + 1], geo.y[:j + 1])
        z_now = np.interp(tq, t[:j + 1], geo.z[:j + 1])

        if trail_length is None:
            start = 0
        else:
            start = max(0, j - int(trail_length))

        line.set_data(geo.x[start:j + 1], geo.y[start:j + 1])
        line.set_3d_properties(geo.z[start:j + 1])

        particle.set_data([x_now], [y_now])
        particle.set_3d_properties([z_now])

        return line, particle
    
    escena.style_axes(ax, title=None, elev=elev, azim=azim, grid=False)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    animation = FuncAnimation(fig, update, frames=len(timeline), interval=interval, blit=False, repeat=True)
    return animation

def main(save=False, ani=False):
    metric = Kerr(M=M, a=a)
    embedding = Embedding(metric, coordenadas=(1, 3), fijas={0: 0.0, 2: np.pi / 2})
    escena = Visualizador2D(embedding, qmin=R_MIN, qmax=R_MAX, nq=NQ, nphi=NPHI, cmap="inferno_r")
    resultado = construir_geodesica(metric, R0, VR, VPHI)
    escena.add_geodesica(resultado, nombre="Geodésica", color="white", linewidth=0.8, particle_size=2)
    campo = agregar_frame_dragging(embedding, R_MIN + 0.15, R_MAX, nr=10, nphi=20)

    r_plus = M + np.sqrt(M**2 - a**2)
    rho_h, z_h = embedding.profile(np.array([r_plus]))
    rho_h = rho_h[0]
    z_h = z_h[0]
    phi_h = np.linspace(0, 2 * np.pi, 200)
    x_h = rho_h * np.cos(phi_h)
    y_h = rho_h * np.sin(phi_h)
    z_h_line = np.full_like(phi_h, z_h)

    fig, ax = escena.draw(title=None, surface_alpha=0.38, elev=30, azim=-55, show_legend=False, show=False)
    dibujar_frame_dragging(ax, campo, fase=0.0, arrow_scale=1.5)
    ax.plot(x_h, y_h, z_h_line, linewidth=0.5, color="white", alpha=0.5)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_axis_off()

    if ani:
        animacion = animar_kerr(escena, campo, frames=steps, interval=30, trail_length=30, elev=30, azim=-55)
        ax.plot(x_h, y_h, z_h_line, linewidth=0.5, color="white", alpha=0.85)

        if save:
            output = ROOT / "Plots" / "graficos" / "kerr_frame_dragging.mp4"
            output.parent.mkdir(parents=True, exist_ok=True)
            animacion.save(output, writer="ffmpeg", fps=30, dpi=100, bitrate=8000)
            print(f"Animación guardada en: {output}")
            plt.show()
            return

    if save:
        output = ROOT / "Plots" / "graficos" / "kerr_frame_dragging.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="black")
        print(f"Guardado en: {output}")

    plt.show()


if __name__ == "__main__":
    main(save=False, ani=True)