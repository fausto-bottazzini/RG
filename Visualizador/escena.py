"""Visualización 3D de embeddings y geodésicas.

La escena consume únicamente objetos ya calculados por el framework:
- ``Embedding`` para la geometría de la superficie.
- resultados de ``SolverGeodesica`` para las trayectorias.

No modifica ni recalcula la dinámica de las geodésicas.
"""

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection


@dataclass
class _GeodesicaVisual:
    resultado: object
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    nombre: str
    color: object
    linewidth: float
    particle_size: float


class Visualizador3D:
    """Escena 3D para un embedding y una o varias geodésicas.

    Parameters
    ----------
    embedding : Embedding
        Objeto de ``Visualizador.embedding``.
    qmin, qmax : float
        Intervalo radial/coordenada de la superficie.
    nq, nphi : int
        Resolución de la superficie.
    figsize : tuple
        Tamaño de la figura.
    """

    def __init__(self, embedding, qmin, qmax, *, nq=300, nphi=160,
                 figsize=(11, 9), cmap="viridis"):
        self.embedding = embedding
        self.qmin = float(qmin)
        self.qmax = float(qmax)
        self.nq = int(nq)
        self.nphi = int(nphi)
        self.figsize = figsize
        self.cmap = cmap

        if self.qmin >= self.qmax:
            raise ValueError("qmin debe ser menor que qmax.")
        if self.nq < 2 or self.nphi < 4:
            raise ValueError("La resolución de la superficie es demasiado baja.")

        self._surface = None
        self._geodesicas = []
        self._fig = None
        self._ax = None

    def _build_surface(self):
        if self._surface is None:
            self._surface = self.embedding.surface(
                self.qmin, self.qmax, nq=self.nq, nphi=self.nphi
            )
        return self._surface

    def _ensure_axes(self, ax=None):
        if ax is not None:
            self._ax = ax
            self._fig = ax.figure
            return self._fig, self._ax

        if self._fig is None or self._ax is None:
            self._fig = plt.figure(figsize=self.figsize, constrained_layout=True)
            self._ax = self._fig.add_subplot(111, projection="3d")
        return self._fig, self._ax

    def add_geodesica(self, resultado, *, nombre=None, color=None,
                      linewidth=2.2, particle_size=55):
        """Agrega una geodésica a la escena.

        La trayectoria se proyecta una sola vez. A partir de ese momento
        la escena reutiliza esas coordenadas, evitando recalcular la métrica.
        """
        if not getattr(resultado, "success", True):
            raise ValueError("El resultado de la geodésica no fue exitoso.")

        x, y, z = self.embedding.project(resultado)
        if len(x) < 2:
            raise ValueError("La geodésica debe contener al menos dos puntos.")

        if nombre is None:
            nombre = f"Geodésica {len(self._geodesicas) + 1}"

        self._geodesicas.append(
            _GeodesicaVisual(
                resultado=resultado,
                x=np.asarray(x, dtype=float),
                y=np.asarray(y, dtype=float),
                z=np.asarray(z, dtype=float),
                nombre=str(nombre),
                color=color,
                linewidth=float(linewidth),
                particle_size=float(particle_size),
            )
        )
        return self

    def clear_geodesicas(self):
        """Elimina todas las trayectorias de la escena."""
        self._geodesicas.clear()
        return self

    def _colors(self):
        cmap = plt.get_cmap("tab10")
        used = set()
        for i, geo in enumerate(self._geodesicas):
            if geo.color is None:
                color = cmap(i % 10)
            else:
                color = geo.color
            used.add(color)
            yield color

    def _style_axes(self, ax, *, title=None, equal=True, grid=False,
                    labels=("X", "Y", "Z")):
        ax.set_xlabel(labels[0], labelpad=8)
        ax.set_ylabel(labels[1], labelpad=8)
        ax.set_zlabel(labels[2], labelpad=8)
        if title is not None:
            ax.set_title(title, pad=18, fontsize=16, fontweight="bold")

        ax.grid(grid)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        if equal:
            X, Y, Z = self._surface
            ax.set_box_aspect((np.ptp(X), np.ptp(Y), max(np.ptp(Z), 1e-12)))

    def draw(self, *, ax=None, title="Embedding isométrico",
             surface_alpha=0.72, surface_linewidth=0,
             equal=True, grid=False, show_legend=True,
             elev=28, azim=-55, show=True):
        """Dibuja una escena estática lista para presentación/publicación."""
        X, Y, Z = self._build_surface()
        fig, ax = self._ensure_axes(ax)

        ax.plot_surface(
            X, Y, Z,
            cmap=self.cmap,
            alpha=surface_alpha,
            linewidth=surface_linewidth,
            antialiased=True,
            rcount=min(self.nq, 250),
            ccount=min(self.nphi, 160),
        )

        handles = []
        for geo, color in zip(self._geodesicas, self._colors()):
            line, = ax.plot(
                geo.x, geo.y, geo.z,
                color=color,
                linewidth=geo.linewidth,
                label=geo.nombre,
                solid_capstyle="round",
            )
            ax.scatter(
                [geo.x[0]], [geo.y[0]], [geo.z[0]],
                color=color, s=geo.particle_size, edgecolor="white",
                linewidth=0.8, depthshade=True, zorder=10,
            )
            handles.append(line)

        self._style_axes(ax, title=title, equal=equal, grid=grid)
        ax.view_init(elev=elev, azim=azim)

        if show_legend and handles:
            ax.legend(loc="upper left", frameon=True, framealpha=0.9)

        if show:
            plt.show()
        return fig, ax

    def animate(self, *, interval=30, frames=500, trail_length=None,
                title="Geodésicas en el embedding", elev=28, azim=-55,
                surface_alpha=0.72, show_legend=True, repeat=True,
                show=True):
        """Crea una animación local con partículas y estelas.

        ``frames`` determina la resolución temporal visual y no modifica
        la integración física original. Las trayectorias se interpolan
        linealmente únicamente para sincronizar la reproducción.
        """
        if not self._geodesicas:
            raise ValueError("No hay geodésicas para animar.")
        if frames < 2:
            raise ValueError("frames debe ser mayor que uno.")

        X, Y, Z = self._build_surface()
        fig, ax = self._ensure_axes()

        ax.plot_surface(
            X, Y, Z,
            cmap=self.cmap,
            alpha=surface_alpha,
            linewidth=0,
            antialiased=True,
            rcount=min(self.nq, 220),
            ccount=min(self.nphi, 140),
        )

        colors = list(self._colors())
        artists = []
        data = []

        # Usamos el parámetro de integración real del resultado.
        for geo in self._geodesicas:
            t = np.asarray(getattr(geo.resultado, "t", np.arange(len(geo.x))), dtype=float)
            if len(t) != len(geo.x):
                t = np.linspace(0.0, 1.0, len(geo.x))
            data.append((t, geo.x, geo.y, geo.z))

            line, = ax.plot([], [], [], color=colors[len(artists)],
                            linewidth=geo.linewidth, label=geo.nombre)
            particle, = ax.plot([], [], [], marker="o", linestyle="",
                                color=colors[len(artists)],
                                markeredgecolor="white", markeredgewidth=0.8,
                                markersize=max(4.0, np.sqrt(geo.particle_size)))
            artists.append((line, particle))

        t0 = min(item[0][0] for item in data)
        tf = max(item[0][-1] for item in data)
        timeline = np.linspace(t0, tf, int(frames))

        def update(frame):
            current = timeline[frame]
            updated = []

            for (t, x, y, z), (line, particle) in zip(data, artists):
                if current <= t[0]:
                    j = 1
                    tq = current
                elif current >= t[-1]:
                    j = len(t) - 1
                    tq = t[-1]
                else:
                    j = int(np.searchsorted(t, current, side="right"))
                    tq = current

                x_now = np.interp(tq, t[:j + 1], x[:j + 1])
                y_now = np.interp(tq, t[:j + 1], y[:j + 1])
                z_now = np.interp(tq, t[:j + 1], z[:j + 1])

                if trail_length is None:
                    start = 0
                else:
                    start = max(0, j - int(trail_length))

                line.set_data(x[start:j + 1], y[start:j + 1])
                line.set_3d_properties(z[start:j + 1])
                particle.set_data([x_now], [y_now])
                particle.set_3d_properties([z_now])
                updated.extend((line, particle))

            return updated

        self._style_axes(ax, title=title, equal=True, grid=False)
        ax.view_init(elev=elev, azim=azim)
        if show_legend:
            ax.legend(loc="upper left", frameon=True, framealpha=0.9)

        animation = FuncAnimation(
            fig, update, frames=len(timeline), interval=interval,
            blit=False, repeat=repeat,
        )
        self._animation = animation

        if show:
            plt.show()
        return animation

    def save(self, filename, *, dpi=300, **kwargs):
        """Guarda la figura estática actual."""
        if self._fig is None:
            self.draw(show=False, **kwargs)
        self._fig.savefig(filename, dpi=dpi, bbox_inches="tight")

    def save_animation(self, filename, *, fps=30, writer=None, dpi=180, **kwargs):
        """Genera y guarda una animación (MP4/GIF según extensión/writer)."""
        animation = self.animate(show=False, **kwargs)
        if writer is None:
            if str(filename).lower().endswith(".gif"):
                writer = "pillow"
            else:
                writer = "ffmpeg"
        animation.save(filename, writer=writer, fps=fps, dpi=dpi)
        return filename
