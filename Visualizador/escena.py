from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import PchipInterpolator

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


class Visualizador2D:
    """Escena 3D para un embedding y una o varias geodésicas."""

    def __init__(self, embedding, qmin, qmax, *, nq=300, nphi=160, figsize=(11, 9), cmap="viridis"):
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
        self._animation = None

    def build_surface(self):
        """Construye la superficie."""
        if self._surface is None:
            self._surface = self.embedding.surface(self.qmin, self.qmax, nq=self.nq, nphi=self.nphi)
        return self._surface

    def ensure_axes(self, ax=None):
        if ax is not None:
            self._ax = ax
            self._fig = ax.figure
            return self._fig, self._ax

        if self._fig is None or self._ax is None:
            self._fig = plt.figure(figsize=self.figsize)
            self._ax = self._fig.add_subplot(111, projection="3d")
        return self._fig, self._ax

    def add_geodesica(self, resultado, *, nombre=None, color=None, linewidth=2.2, particle_size=55):
        """Agrega una geodésica a la escena.
        La trayectoria se proyecta una sola vez. 
        A partir de ese momento la escena reutiliza esas coordenadas, evitando recalcular la métrica.
        """
        self.build_surface()

        if not hasattr(resultado, "y"):
            raise ValueError("resultado debe ser un resultado de scipy.solve_ivp.")
        if resultado.y.shape[1] < 2:
            raise ValueError("La geodésica debe contener al menos dos puntos.")

        x, y, z = self.embedding.project(resultado)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        if len(x) < 2:
            raise ValueError("La parte proyectable de la geodésica tiene menos de dos puntos.")

        if nombre is None:
            nombre = f"Geodésica {len(self._geodesicas) + 1}"

        geo = _GeodesicaVisual(
            resultado=resultado,
            x=x,
            y=y,
            z=z,
            nombre=str(nombre),
            color=color,
            linewidth=float(linewidth),
            particle_size=float(particle_size),
        )

        self._geodesicas.append(geo)

        return geo

    @property
    def geodesicas(self):
        return tuple(self._geodesicas)

    def clear_geodesicas(self):
        """Elimina todas las trayectorias de la escena."""
        self._geodesicas.clear()
        return self

    def _colors(self):
        cmap = plt.get_cmap("tab20b")
        for i, geo in enumerate(self._geodesicas):
            if geo.color is None:
                yield cmap(i % 10)
            else:
                yield geo.color

    def style_axes(self, ax, *, title=None, elev=28, azim=-55, grid=False):
        ax.set_xlabel("X", labelpad=8)
        ax.set_ylabel("Y", labelpad=8)
        ax.set_zlabel("Z", labelpad=8)
        if title is not None:
            ax.set_title(title, pad=18, fontsize=16, fontweight="bold")

        ax.grid(grid)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        X, Y, Z = self._surface
        ax.set_box_aspect((np.ptp(X), np.ptp(Y), max(np.ptp(Z), 1e-12)))
        ax.view_init(elev=elev, azim=azim)

    @staticmethod
    def _get_parameter(geo):
        """Obtiene el parámetro de integración de una geodésica."""
        x = np.asarray(geo.x, dtype=float)
        y = np.asarray(geo.y, dtype=float)
        z = np.asarray(geo.z, dtype=float)
        t = getattr(geo.resultado, "t", None)

        if t is None:
            t = np.arange(len(x), dtype=float)
        else:
            t = np.asarray(t, dtype=float)

        if len(t) == len(x):
            dt = np.diff(t)
            mask = np.concatenate(([True], dt > 0.0))
            return (t[mask], x[mask], y[mask], z[mask])

        s = np.zeros(len(x), dtype=float)
        if len(x) > 1:
            ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
            s[1:] = np.cumsum(ds)
        return (s,x,y,z)

    @staticmethod
    def _interpolar_geodesica(t,x,y,z,timeline):
        """Interpola una geodesica sobre un timeline común dado."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        t = np.asarray(t, dtype=float)    

        if len(t) == 0:
            raise ValueError("La geodesica no contiene puntos proyectados.")

        dt = np.diff(t)
        mask = np.concatenate(([True], dt > 0.0))
        t = t[mask]
        x = x[mask]
        y = y[mask]
        z = z[mask]

        if len(t) == 1:
            x_visual = np.full_like(timeline, x[0], dtype=float)
            y_visual = np.full_like(timeline, y[0], dtype=float)
            z_visual = np.full_like(timeline, z[0], dtype=float)
            return (x_visual, y_visual, z_visual)

        inside = (timeline >= t[0]) & (timeline <= t[-1])
        x_visual = np.empty_like(timeline, dtype=float)
        y_visual = np.empty_like(timeline, dtype=float)
        z_visual = np.empty_like(timeline, dtype=float)

        if len(t) >= 3:
            fx = PchipInterpolator(t, x)
            fy = PchipInterpolator(t, y)
            fz = PchipInterpolator(t, z)
            x_visual[inside] = fx(timeline[inside])
            y_visual[inside] = fy(timeline[inside])
            z_visual[inside] = fz(timeline[inside])

        else:
            x_visual[inside] = np.interp(timeline[inside], t, x)
            y_visual[inside] = np.interp(timeline[inside], t, y)
            z_visual[inside] = np.interp(timeline[inside], t, z)

        before = timeline < t[0]
        x_visual[before] = x[0]
        y_visual[before] = y[0]
        z_visual[before] = z[0]

        after = timeline > t[-1]
        x_visual[after] = x[-1]
        y_visual[after] = y[-1]
        z_visual[after] = z[-1]

        return (x_visual, y_visual, z_visual)

    def _pre_animacion(self, frames):
        """Construye un timeline común para toda la escena."""
        trayectorias = []
        for geo in self._geodesicas:
            trayectorias.append(self._get_parameter(geo))

        t_start = min(t[0] for t , _, _ , _ in trayectorias)
        t_end = max(t[-1] for t, _, _, _ in trayectorias)
        timeline = np.linspace(t_start, t_end, int(frames))

        visual_data = []
        for t,x,y,z in trayectorias:
            x_visual, y_visual, z_visual = (self._interpolar_geodesica(t,x,y,z,timeline))
            visual_data.append((x_visual, y_visual, z_visual))

        return (timeline, visual_data)

    def draw(self, *, ax=None, title="Embedding isométrico", surface_alpha=0.72, grid=False, show_legend=True, elev=28, azim=-55, show=True):
        """Dibuja una escena estática lista para presentación."""
        self.build_surface()
        X, Y, Z = self._surface
        fig, ax = self.ensure_axes(ax)

        ax.plot_surface(
            X, Y, Z,
            cmap=self.cmap,
            alpha=surface_alpha,
            linewidth=0,
            antialiased=True,
            rcount=min(self.nq, 250),
            ccount=min(self.nphi, 160),
            zorder = 0
        )

        handles = []
        for geo, color in zip(self._geodesicas, self._colors()):
            line, = ax.plot(geo.x, geo.y, geo.z, color=color, linewidth=geo.linewidth, label=geo.nombre, solid_capstyle="round", zorder=13)
            ax.scatter([geo.x[0]], [geo.y[0]], [geo.z[0]], color=color, s=geo.particle_size, edgecolor="white", linewidth=1.2, depthshade=True, zorder=15)
            ax.scatter([geo.x[-1]], [geo.y[-1]], [geo.z[-1]], color=color, s=geo.particle_size*0.65, edgecolor="white", linewidth=1.2, depthshade=True, zorder=15)
            handles.append(line)

        self.style_axes(ax, title=title, elev=elev, azim=azim, grid=grid)

        if show_legend and handles:
            ax.legend(loc="upper left", frameon=True, framealpha=0.9)

        if show:
            plt.show()
        return fig, ax

    def animate(self, *, interval=30, frames=600, trail_length=None, title="Geodésicas en el embedding", elev=28, azim=-55, surface_alpha=0.72, show_legend=True, repeat=True, show=True):
        """Crea una animación con partículas y estelas.
        frames determina la resolución temporal visual y no modifica la integración física original. 
        """
        if not self._geodesicas:
            raise ValueError("No hay geodésicas para animar.")
        if frames < 2:
            raise ValueError("frames debe ser mayor que uno.")

        self.build_surface()
        X, Y, Z = self._surface
        fig, ax = self.ensure_axes()

        ax.plot_surface(
            X, Y, Z,
            cmap=self.cmap,
            alpha=surface_alpha,
            linewidth=0,
            antialiased=True,
            rcount=min(self.nq, 220),
            ccount=min(self.nphi, 140),
            zorder=0
        )

        timeline, visual_data = (self._pre_animacion(frames))
        colors = list(self._colors())
        artists = []

        for i, geo in enumerate(self._geodesicas):
            color = colors[i]
            line, = ax.plot([], [], [], color=color, linewidth=geo.linewidth, label=geo.nombre, solid_capstyle="round", zorder=15)
            particle, = ax.plot([], [], [], marker="o", linestyle="", color=color, markeredgecolor="white", markeredgewidth=0.7, markersize=max(4.0, np.sqrt(geo.particle_size)), zorder=20)
            artists.append((line, particle))

        def update(frame):
            updated = []
            for (x, y, z), (line, particle) in zip(visual_data, artists):
                if trail_length is None:
                    start = 0
                else:
                    start = max(0, frame - int(trail_length) + 1)

                xs = x[start:frame + 1]
                ys = y[start:frame + 1]
                zs = z[start:frame + 1]

                end = frame + 1
                line.set_data(xs, ys)
                line.set_3d_properties(zs)

                particle.set_data([x[frame]], [y[frame]])
                particle.set_3d_properties([z[frame]])

                updated.extend([line, particle])

            return updated

        self.style_axes(ax, title=title, elev=elev, azim=azim, grid=False)
        if show_legend:
            ax.legend(loc="upper left", frameon=True, framealpha=0.9)

        animation = FuncAnimation(fig, update, frames=len(timeline), interval=interval, blit=False, repeat=repeat, cache_frame_data=False)
        self._animation = animation

        if show:
            plt.show()
        return animation

    def save(self, filename, *, dpi=300, **kwargs):
        """Guarda la figura estática actual."""
        if self._fig is None:
            self.draw(show=False, **kwargs)
        self._fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        print(f"{filename} - save")

    def save_animation(self, filename, *, fps=30, writer=None, dpi=180, bitrate=8000, **kwargs):
        """Genera y guarda una animación (MP4/GIF según extensión/writer)."""
        animation = self.animate(show=False, **kwargs)
        if writer is None:
            if str(filename).lower().endswith(".gif"):
                from matplotlib.animation import PillowWriter
                writer = PillowWriter(fps=fps)
            else:
                from matplotlib.animation import FFMpegWriter
                writer = FFMpegWriter(fps=fps, bitrate=bitrate)

        animation.save(filename, writer=writer, dpi=dpi)

        print(f"{filename} - animation save")
        return filename