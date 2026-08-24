import time
import numpy as np

from Metricas.schwarzschild import Schwarzschild

from Solver.ray_tracing import (
    SolverRayTracing,
    STATUS_ACTIVE,
    STATUS_ESCAPE,
    STATUS_HORIZON,
    STATUS_MAX_LAMBDA,
    EscapeEvent,
    HorizonEvent,
)

from Tensores.operaciones import tetrada_obs, transformar_vector
from Visualizador.RayTracing.cam import Camara


# ============================================================
# CONFIGURACIÓN
# ============================================================

M = 1.0

x0 = np.array([
    0.0,
    10.0,
    np.pi / 2,
    0.0,
])

RESOLUCION = (100, 100)
FOV = 60.0

RTOL = 1e-9
ATOL = 1e-11

LAMBDA_MAX = 100.0

H0 = 0.01
H_MIN = 1e-10
H_MAX = 0.1

R_STOP = 2.001
R_MAX = 20.0


# ============================================================
# CÁMARA
# ============================================================

def construir_rayones():

    metrica = Schwarzschild(M=M)

    metric_num = metrica.numeric("metric")
    valores = metric_num.evaluar_valores(*x0)

    g0 = np.zeros((4, 4), dtype=float)

    for idx, valor in zip(metric_num.indices, valores):
        g0[idx] = valor

    u_obs = np.zeros(4)
    u_obs[0] = 1.0 / np.sqrt(-g0[0, 0])

    e = tetrada_obs(
        metrica,
        x0,
        u_obs,
    )

    camara = Camara(
        posicion=x0,
        resolucion=RESOLUCION,
        fov=np.radians(FOV),
        foward=(-1.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
    )

    k_local = camara.rays_local()

    k = transformar_vector(
        e,
        k_local,
    )

    N = len(k)

    y0 = np.empty(
        (N, 8),
        dtype=float,
    )

    y0[:, :4] = x0
    y0[:, 4:] = k

    return metrica, y0


# ============================================================
# EVENTOS
# ============================================================

def construir_eventos():

    return (
        HorizonEvent(
            R_STOP,
            radial_index=1,
        ),
        EscapeEvent(
            R_MAX,
            radial_index=1,
        ),
    )


# ============================================================
# SOLVER INSTRUMENTADO
# ============================================================

def resolver_instrumentado(
    solver,
    y0,
    eventos,
):
    """
    Replica el solver actual y mide:

      - número de rayos activos por iteración
      - tiempo por iteración
      - tiempo acumulado por tamaño de batch
      - número de llamadas _rk45
    """

    y = np.asarray(
        y0,
        dtype=float,
    ).copy()

    N = len(y)

    parametro = np.zeros(N)

    h = np.full(
        N,
        H0,
        dtype=float,
    )

    status = np.full(
        N,
        STATUS_ACTIVE,
        dtype=np.int8,
    )

    pasos = np.zeros(
        N,
        dtype=int,
    )

    rechazos = np.zeros(
        N,
        dtype=int,
    )

    # --------------------------------------------------------
    # Instrumentación
    # --------------------------------------------------------

    activos_hist = []
    tiempos_iter = []

    llamadas_rk45 = 0

    inicio_total = time.perf_counter()

    # --------------------------------------------------------
    # Integración
    # --------------------------------------------------------

    for _ in range(1_000_000):

        inicio_iter = time.perf_counter()

        activos = status == STATUS_ACTIVE

        if not np.any(activos):
            break

        indices = np.flatnonzero(
            activos
        )

        n_activos = len(indices)

        activos_hist.append(
            n_activos
        )

        y_act = y[indices]
        h_act = h[indices]

        restante = (
            LAMBDA_MAX
            - parametro[indices]
        )

        h_act = np.minimum(
            h_act,
            restante,
        )

        y_new, error = solver._rk45(
            y_act,
            h_act,
        )

        llamadas_rk45 += 1

        aceptado = error <= 1.0

        # ----------------------------------------------------
        # RECHAZADOS
        # ----------------------------------------------------

        if np.any(~aceptado):

            idx = indices[~aceptado]

            factor = solver._factor_paso(
                error[~aceptado],
                np.zeros(
                    np.sum(~aceptado),
                    dtype=bool,
                ),
            )

            h[idx] *= factor

            h[idx] = np.maximum(
                h[idx],
                H_MIN,
            )

            rechazos[idx] += 1

        # ----------------------------------------------------
        # ACEPTADOS
        # ----------------------------------------------------

        if np.any(aceptado):

            local = indices[aceptado]

            y_prev = y_act[aceptado]
            y_next = y_new[aceptado]

            h_next = h_act[aceptado]
            error_next = error[aceptado]

            evento_alpha = np.full(
                len(local),
                np.inf,
                dtype=float,
            )

            evento_code = np.full(
                len(local),
                -1,
                dtype=np.int8,
            )

            for evento in eventos:

                detectado, alpha = evento.detect(
                    y_prev,
                    y_next,
                )

                tomar = (
                    detectado
                    & (alpha < evento_alpha)
                )

                evento_alpha[tomar] = alpha[tomar]
                evento_code[tomar] = evento.code

            ocurrio = np.isfinite(
                evento_alpha
            )

            normales = ~ocurrio

            # -----------------------------------------------
            # NORMALES
            # -----------------------------------------------

            if np.any(normales):

                idx = local[normales]

                y[idx] = y_next[normales]

                parametro[idx] += (
                    h_next[normales]
                )

                pasos[idx] += 1

            # -----------------------------------------------
            # EVENTOS
            # -----------------------------------------------

            if np.any(ocurrio):

                idx = local[ocurrio]

                alpha = evento_alpha[
                    ocurrio
                ]

                y_hit = (
                    y_prev[ocurrio]
                    + alpha[:, None]
                    * (
                        y_next[ocurrio]
                        - y_prev[ocurrio]
                    )
                )

                y[idx] = y_hit

                parametro[idx] += (
                    alpha
                    * h_next[ocurrio]
                )

                status[idx] = (
                    evento_code[
                        ocurrio
                    ]
                )

                pasos[idx] += 1

            # -----------------------------------------------
            # ADAPTACIÓN DE h
            # -----------------------------------------------

            if np.any(normales):

                factor = solver._factor_paso(
                    error_next[normales],
                    np.ones(
                        np.sum(normales),
                        dtype=bool,
                    ),
                )

                idx = local[normales]

                h[idx] *= factor

                h[idx] = np.clip(
                    h[idx],
                    H_MIN,
                    H_MAX,
                )

            # -----------------------------------------------
            # LAMBDA MAX
            # -----------------------------------------------

            llego = (
                normales
                & (
                    parametro[local]
                    >= (
                        LAMBDA_MAX
                        - np.finfo(float).eps
                    )
                )
            )

            if np.any(llego):

                status[
                    local[llego]
                ] = STATUS_MAX_LAMBDA

        tiempos_iter.append(
            time.perf_counter()
            - inicio_iter
        )

    tiempo_total = (
        time.perf_counter()
        - inicio_total
    )

    return {
        "status": status,
        "pasos": pasos,
        "rechazos": rechazos,
        "tiempo": tiempo_total,
        "activos": np.asarray(
            activos_hist,
            dtype=int,
        ),
        "tiempos_iter": np.asarray(
            tiempos_iter,
            dtype=float,
        ),
        "llamadas_rk45": llamadas_rk45,
    }


# ============================================================
# ANÁLISIS POR TAMAÑO DE BATCH
# ============================================================

def analizar_rangos(
    activos,
    tiempos,
):
    """
    Agrupa las iteraciones según cantidad de rayos activos.
    """

    rangos = (
        ("10000-5001", 5001, np.inf),
        ("5000-2001", 2001, 5000),
        ("2000-1001", 1001, 2000),
        ("1000-501", 501, 1000),
        ("500-101", 101, 500),
        ("100-51", 51, 100),
        ("50-11", 11, 50),
        ("10-1", 1, 10),
    )

    print()
    print("=" * 90)
    print("COSTO SEGÚN CANTIDAD DE RAYOS ACTIVOS")
    print("=" * 90)

    print(
        f"{'rango':>14} "
        f"{'iteraciones':>14} "
        f"{'tiempo [s]':>14} "
        f"{'% tiempo':>12} "
        f"{'rayos prom.':>14} "
        f"{'us/iter':>14}"
    )

    print("-" * 90)

    total = tiempos.sum()

    for nombre, minimo, maximo in rangos:

        mask = (
            (activos >= minimo)
            & (activos <= maximo)
        )

        if not np.any(mask):
            continue

        t = tiempos[mask].sum()
        n = np.count_nonzero(mask)

        promedio_activos = activos[
            mask
        ].mean()

        us_iter = (
            t / n * 1e6
        )

        print(
            f"{nombre:>14} "
            f"{n:14d} "
            f"{t:14.6f} "
            f"{100.0 * t / total:12.3f} "
            f"{promedio_activos:14.2f} "
            f"{us_iter:14.3f}"
        )


# ============================================================
# ESTADÍSTICAS ÚTILES
# ============================================================

def imprimir_estadisticas(
    resultado,
):

    activos = resultado["activos"]
    tiempos = resultado["tiempos_iter"]

    print()
    print("=" * 90)
    print("ESTADÍSTICAS DE BATCH")
    print("=" * 90)

    print(
        f"iteraciones totales     = "
        f"{len(activos)}"
    )

    print(
        f"rayos activos iniciales = "
        f"{activos[0]}"
    )

    print(
        f"rayos activos finales   = "
        f"{activos[-1]}"
    )

    print(
        f"batch medio             = "
        f"{activos.mean():.2f}"
    )

    print(
        f"batch mediano           = "
        f"{np.median(activos):.2f}"
    )

    print(
        f"batch mínimo            = "
        f"{activos.min()}"
    )

    print(
        f"batch máximo            = "
        f"{activos.max()}"
    )

    # Tiempo por iteración

    print()
    print("TIEMPO POR ITERACIÓN")
    print("-" * 60)

    print(
        f"mínimo   = "
        f"{tiempos.min() * 1e6:.3f} us"
    )

    print(
        f"mediana  = "
        f"{np.median(tiempos) * 1e6:.3f} us"
    )

    print(
        f"máximo   = "
        f"{tiempos.max() * 1e6:.3f} us"
    )

    # Últimas etapas

    print()
    print("ÚLTIMAS ITERACIONES")
    print("-" * 60)

    n = min(
        20,
        len(activos),
    )

    for i in range(
        len(activos) - n,
        len(activos),
    ):

        print(
            f"iter {i:4d} : "
            f"{activos[i]:5d} rayos   "
            f"{tiempos[i] * 1e6:10.3f} us"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("BENCHMARK — COSTO DE LOS BATCHES PEQUEÑOS")
    print("=" * 90)

    metrica, y0 = construir_rayones()

    eventos = construir_eventos()

    solver = SolverRayTracing(
        metrica,
        rtol=RTOL,
        atol=ATOL,
    )

    resultado = resolver_instrumentado(
        solver,
        y0,
        eventos,
    )

    N = len(y0)

    print()
    print("RESULTADO")
    print("-" * 90)

    print(
        f"rayos               = {N}"
    )

    print(
        f"tiempo total        = "
        f"{resultado['tiempo']:.6f} s"
    )

    print(
        f"us/rayo             = "
        f"{resultado['tiempo'] / N * 1e6:.3f}"
    )

    print(
        f"llamadas _rk45      = "
        f"{resultado['llamadas_rk45']}"
    )

    print(
        f"pasos promedio      = "
        f"{resultado['pasos'].mean():.3f}"
    )

    print(
        f"rechazos totales    = "
        f"{resultado['rechazos'].sum()}"
    )

    # --------------------------------------------------------
    # Estados
    # --------------------------------------------------------

    print()
    print("ESTADOS")
    print("-" * 60)

    for codigo, nombre in (
        (STATUS_ESCAPE, "ESCAPE"),
        (STATUS_HORIZON, "HORIZON"),
        (STATUS_MAX_LAMBDA, "MAX_LAMBDA"),
    ):

        n = np.count_nonzero(
            resultado["status"] == codigo
        )

        if n == 0:
            continue

        print(
            f"{nombre:>12}: "
            f"{n:6d} "
            f"({100.0 * n / N:7.3f} %)"
        )

    # --------------------------------------------------------
    # Estadísticas
    # --------------------------------------------------------

    imprimir_estadisticas(
        resultado
    )

    analizar_rangos(
        resultado["activos"],
        resultado["tiempos_iter"],
    )


if __name__ == "__main__":
    main()