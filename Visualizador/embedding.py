import numpy as np
import matplotlib.pyplot as plt

class Embedding:
    """
    Embedding espacial de métricas.

    Actualmente implementa únicamente el embedding de Flamm
    para la métrica de Schwarzschild.
    """

    def __init__(self, metrica):
        self.metrica = metrica

        if "M" not in metrica.params:
            raise NotImplementedError("Embedding disponible únicamente para Schwarzschild.")

        try:
            self.M = float(metrica.params["M"])
        except Exception:
            raise ValueError("El embedding requiere parámetros numéricos.")

    def surface(self, rmax=20, nr=100, nphi=100):
        """
        Devuelve las coordenadas (X,Y,Z) del embedding de Flamm.
        """
        rmin = 2*self.M*1.001
        r = np.linspace(rmin, rmax, nr)
        phi = np.linspace(0, 2*np.pi, nphi)
        R, Phi = np.meshgrid(r, phi)
        X = R*np.cos(Phi)
        Y = R*np.sin(Phi)
        Z = 2*np.sqrt(2*self.M*(R-2*self.M))

        return X, Y, Z

    def plot(self, rmax=20, nr=250, nphi=250, cmap="viridis", alpha=0.9, ax=None):
        """
        Grafica el embedding.
        """

        X, Y, Z = self.surface(rmax, nr, nphi)

        if ax is None:
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111, projection="3d")

        else:
            fig = ax.figure

        ax.plot_surface(X,Y,Z, cmap=cmap, linewidth=0, antialiased=True, alpha=alpha)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.set_title("Embedding de Flamm")
        ax.set_box_aspect((1,1,0.45))

        return fig, ax