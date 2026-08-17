import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

class Embedding:
    def __init__(self, metrica, coordenadas=(1, 3), fijas=None):
        self.metrica = metrica
        self.q_index = coordenadas[0]
        self.phi_index = coordenadas[1]
        self.dim = metrica.g.dim

        if self.q_index == self.phi_index:
            raise ValueError("Las dos coordenadas de la sección deben ser diferentes.")
        if not (0 <= self.q_index < self.dim and 0 <= self.phi_index < self.dim):
            raise IndexError("Índice de coordenada fuera de rango.")

        self.fijas = {} if fijas is None else dict(fijas)
        self._validar_fijas()
        self._metric_num = metrica.numeric("metric")
        self._metric_indices = {indice: i for i, indice in enumerate(self._metric_num.indices)}
        self._validar_componentes()

    def _validar_fijas(self):
        """
        Verifica que todas las coordenadas no pertenecientes
        a la sección estén fijadas.
        """

        requeridas = set(range(self.dim))
        requeridas.remove(self.q_index)
        requeridas.remove(self.phi_index)
        faltantes = requeridas - set(self.fijas)

        if faltantes:
            raise ValueError(f"Las siguientes coordenadas deben estar fijadas: {sorted(faltantes)}")

        sobrantes = set(self.fijas) - requeridas

        if sobrantes:
            raise ValueError(f"No se pueden fijar las coordenadas de la sección: {sorted(sobrantes)}")

    def _validar_componentes(self):
        """
        Verifica que los componentes necesarios de la métrica
        estén disponibles.
        """
        necesarios = ((self.q_index, self.q_index), (self.phi_index, self.phi_index))

        for indice in necesarios:
            if indice not in self._metric_indices:
                raise ValueError(f"La métrica no contiene el componente g{indice} necesario para el embedding.")

    def _coords(self, q, phi=0.0):
        """Construye el vector completo de coordenadas."""
        x = np.empty(self.dim, dtype=float)
        x[self.q_index] = q
        x[self.phi_index] = phi
        for indice, valor in self.fijas.items():
            x[indice] = valor
        return x

    def _metric_components(self, q):
        """
        Evalúa los componentes de la métrica necesarios para
        construir el embedding.

        La función numérica de TensorNumerico ya está compilada
        y se reutiliza directamente.
        """

        q = np.asarray(q, dtype=float)
        if q.ndim != 1:
            raise ValueError("q debe ser un array unidimensional.")

        indice_qq = self._metric_indices[(self.q_index, self.q_index)]
        indice_phiphi = self._metric_indices[(self.phi_index, self.phi_index)]
        indice_qphi = self._metric_indices.get((self.q_index, self.phi_index))

        g_qq = np.empty_like(q)
        g_phiphi = np.empty_like(q)
        g_qphi = np.zeros_like(q)

        for i, valor in enumerate(q):
            x = self._coords(valor)
            valores = self._metric_num.evaluar_valores(*x)

            g_qq[i] = valores[indice_qq]
            g_phiphi[i] = valores[indice_phiphi]

            if indice_qphi is not None:
                g_qphi[i] = valores[indice_qphi]

        return g_qq, g_phiphi, g_qphi

    def _rho(self, q):
        """
        Radio de la superficie de revolución:
            rho(q) = sqrt(g_phiphi)
        """
        _, g_phiphi, _ = self._metric_components(q)
        if np.any(g_phiphi <= 0):
            raise ValueError("g_phiphi debe ser positiva para construir el embedding.")
        return np.sqrt(g_phiphi)

    def _embedding_profile(self, q):
        """Construye el perfil meridiano del embedding."""
        q = np.asarray(q, dtype=float)

        if q.ndim != 1:
            raise ValueError("q debe ser un array unidimensional.")
        if len(q) < 2:
            raise ValueError("Se necesitan al menos dos puntos.")
        if np.any(np.diff(q) <= 0):
            raise ValueError("Los valores de q deben ser estrictamente crecientes.")
        if not np.all(np.isfinite(q)):
            raise ValueError("q contiene valores no finitos.")
        if np.any(q <= 0):
            raise ValueError("Los valores de q deben pertenecer a un dominio físicamente válido para la métrica.")

        g_qq, g_phiphi, g_qphi = (self._metric_components(q))
        escala = max(1.0, np.max(np.abs(g_qq)), np.max(np.abs(g_phiphi)))
        tolerancia_cruzada = 1e-10 * escala

        if np.max(np.abs(g_qphi)) > tolerancia_cruzada:
            raise NotImplementedError("La sección tiene un término cruzado g_qphi != 0. " \
                                    "El embedding actual requiere g_qphi = 0.")

        rho = np.sqrt(g_phiphi)
        drho_dq = np.gradient(rho, q)
        radicando = g_qq - drho_dq**2
        tolerancia_radicando = (1e-10 * np.maximum(1.0, np.abs(g_qq)))

        if np.any(radicando < -tolerancia_radicando):
            raise ValueError("El embedding isométrico no existe en todo el intervalo solicitado.")

        radicando = np.maximum(radicando, 0.0)
        dz_dq = np.sqrt(radicando)
        z = cumulative_trapezoid(dz_dq, q, initial=0.0)

        return rho, z

    def surface(self, qmin, qmax, nq=300, nphi=200):
        """Construye la superficie de embedding."""

        q = np.linspace(qmin, qmax, nq)
        rho, z = self._embedding_profile(q)

        self._q_profile = q
        self._rho_profile = rho
        self._z_profile = z

        phi = np.linspace(0.0, 2.0 * np.pi, nphi)
        Phi, Rho = np.meshgrid(phi, rho, indexing="ij")

        Z = np.broadcast_to(z, Rho.shape)
        X = Rho * np.cos(Phi)
        Y = Rho * np.sin(Phi)

        return X, Y, Z

    def project(self, resultado):
        """Proyecta una geodésica obtenida mediante SolverGeodesica."""
        if not hasattr(self, "_q_profile"):
            raise RuntimeError("Debe construirse la superficie antes de proyectar.")

        q = np.asarray(resultado.y[self.q_index], dtype=float)
        phi = np.asarray(resultado.y[self.phi_index], dtype=float)

        if len(q) < 2:
            raise ValueError("La geodésica debe contener al menos dos puntos.")

        q_profile = self._q_profile
        rho_profile = self._rho_profile
        z_profile = self._z_profile

        mascara = ((q >= q_profile[0]) & (q <= q_profile[-1]))
        if not np.any(mascara):
            raise ValueError("La geodésica no interseca el intervalo del embedding.")
        q = q[mascara]
        phi = phi[mascara]

        rho = np.interp(q, q_profile, rho_profile)
        z = np.interp(q, q_profile, z_profile)
        X = rho * np.cos(phi)
        Y = rho * np.sin(phi)

        return X, Y, z

    def plot(self, qmin, qmax, resultado=None, nq=250, nphi=200, ax=None, alpha=0.75, cmap="viridis", linecolor = "black"):
        """Visualiza el embedding y la geodesica."""

        X, Y, Z = self.surface(qmin, qmax, nq=nq, nphi=nphi)

        if ax is None:
            fig = plt.figure(figsize=(9, 8))
            ax = fig.add_subplot(111, projection="3d")

        else:
            fig = ax.figure

        ax.plot_surface(X, Y, Z, cmap=cmap, alpha=alpha, linewidth=0, antialiased=True, zorder=0)

        if resultado is not None:
            x, y, z = self.project(resultado)
            ax.plot(x, y, z, linewidth=2.0, color=linecolor, zorder=3)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Embedding isométrico")
        ax.set_box_aspect((np.ptp(X), np.ptp(Y), max(np.ptp(Z), np.finfo(float).eps)))
        return fig, ax

    def profile(self, q_values):
        """Evalúa el perfil del embedding para los valores de q dados."""
        q_values = np.asarray(q_values, dtype=float)
        rho = np.interp(q_values, self._q_profile, self._rho_profile)
        z = np.interp(q_values, self._q_profile, self._z_profile)
        return rho, z