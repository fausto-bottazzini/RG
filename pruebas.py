from pathlib import Path
import os
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Metricas.schwarzschild import Schwarzschild

from Solver.ray_tracing import (
    SolverRayTracing,
    EscapeEvent,
    HorizonEvent,
    STATUS_ESCAPE,
    STATUS_HORIZON,
    STATUS_MAX_LAMBDA,
    STATUS_STEP_FAILURE,
)

from Tensores.operaciones import (
    tetrada_obs,
    transformar_vector,
)

from Visualizador.RayTracing.cam import Camara


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

M = 1.0

SIZE = int(
    os.environ.get(
        "RT_TEST_SIZE",
        250,
    )
)

RESOLUCION = (
    SIZE,
    SIZE,
)

FOV = float(
    os.environ.get(
        "RT_TEST_FOV",
        60.0,
    )
)

R0 = float(
    os.environ.get(
        "RT_TEST_R0",
        50.0,
    )
)

R_STOP = 2.001
R_ESCAPE = R0

RTOL = 1e-7
ATOL = 1e-9

H0 = 0.01
H_MIN = 1e-10
H_MAX = 0.1

LAMBDA_FACTOR = 10.0

# Tamaño máximo de bloque.
# None = todos los rayos juntos.
BLOQUES = (
    None,
    32768,
    16384,
    8192,
    4096,
    2048,
    1024,
)


# =============================================================================
# CÁMARA
# =============================================================================

def construir_observador(metrica):

    x0 = np.array(
        [
            0.0,
            R0,
            np.pi / 2.0,
            0.0,
        ]
    )

    metric_num = metrica.numeric("metric")
    valores = metric_num.evaluar_valores(*x0)

    g = np.zeros(
        (4, 4),
        dtype=float,
    )

    for idx, valor in zip(
        metric_num.indices,
        valores,
    ):
        g[idx] = valor

    u_obs = np.zeros(
        4,
        dtype=float,
    )

    u_obs[0] = (
        1.0 / np.sqrt(-g[0, 0])
    )

    tetrada = tetrada_obs(
        metrica,
        x0,
        u_obs,
    )

    return x0, tetrada


def construir_rayos(
    x0,
    tetrada,
):

    camara = Camara(
        posicion=x0,
        resolucion=RESOLUCION,
        fov=np.radians(FOV),
        foward=(-1.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
    )

    k_local = camara.rays_local()
    k = transformar_vector(
        tetrada,
        k_local,
    )

    y0 = np.empty(
        (len(k), 8),
        dtype=float,
    )

    y0[:, :4] = x0
    y0[:, 4:] = k

    return camara, y0


# =============================================================================
# DIAGNÓSTICO DE CÁMARA
# =============================================================================

def diagnostico_camara(
    metrica,
    y0,
):

    x = y0[:, :4]
    u = y0[:, 4:]

    metric_num = metrica.numeric("metric")

    valores = metric_num.evaluar_valores(
        *x.T
    )

    mapa = {
        idx: np.asarray(valor)
        for idx, valor in zip(
            metric_num.indices,
            valores,
        )
    }

    gtt = mapa[(0, 0)]
    gphiphi = mapa[(3, 3)]

    energy = -gtt * u[:, 0]
    angular = gphiphi * u[:, 3]

    impact = np.abs(
        angular / energy
    )

    b_critico = (
        3.0
        * np.sqrt(3.0)
        * M
    )

    falling = impact < b_critico

    print()
    print("=" * 90)
    print("DIAGNÓSTICO DE LA CÁMARA")
    print("=" * 90)

    print(
        f"rayos                  = "
        f"{len(y0):,}"
    )

    print(
        f"b crítico              = "
        f"{b_critico:.8f}"
    )

    print(
        f"b min / mediana / max  = "
        f"{impact.min():.8f} / "
        f"{np.median(impact):.8f} / "
        f"{impact.max():.8f}"
    )

    print(
        f"b < b crítico          = "
        f"{falling.sum():,}"
        f" ({100.0 * falling.mean():.2f} %)"
    )


# =============================================================================
# EVENTOS
# =============================================================================

def construir_eventos():

    return (
        HorizonEvent(
            R_STOP,
            radial_index=1,
        ),
        EscapeEvent(
            R_ESCAPE,
            radial_index=1,
        ),
    )


# =============================================================================
# UNA CORRIDA COMPLETA
# =============================================================================

def resolver_completo(
    solver,
    y0,
    eventos,
):

    inicio = time.perf_counter()

    resultado = solver.resolver(
        y0,
        lambda_max=None,
        h0=H0,
        h_min=H_MIN,
        h_max=H_MAX,
        eventos=eventos,
        progress=False,
    )

    elapsed = (
        time.perf_counter()
        - inicio
    )

    return resultado, elapsed


# =============================================================================
# RESOLVER POR BLOQUES
# =============================================================================

def resolver_bloques(
    solver,
    y0,
    eventos,
    bloque,
):

    N = len(y0)

    estado = np.empty_like(y0)
    status = np.empty(
        N,
        dtype=np.int8,
    )
    parametro = np.empty(
        N,
        dtype=float,
    )

    pasos_aceptados = np.empty(
        N,
        dtype=int,
    )

    pasos_rechazados = np.empty(
        N,
        dtype=int,
    )

    inicio = time.perf_counter()

    bloques_procesados = 0

    for inicio_idx in range(
        0,
        N,
        bloque,
    ):

        fin_idx = min(
            inicio_idx + bloque,
            N,
        )

        resultado = solver.resolver(
            y0[inicio_idx:fin_idx],
            lambda_max=None,
            h0=H0,
            h_min=H_MIN,
            h_max=H_MAX,
            eventos=eventos,
            progress=False,
        )

        estado[
            inicio_idx:fin_idx
        ] = resultado.estado

        status[
            inicio_idx:fin_idx
        ] = resultado.status

        parametro[
            inicio_idx:fin_idx
        ] = resultado.parametro

        pasos_aceptados[
            inicio_idx:fin_idx
        ] = resultado.pasos_aceptados

        pasos_rechazados[
            inicio_idx:fin_idx
        ] = resultado.pasos_rechazados

        bloques_procesados += 1

    elapsed = (
        time.perf_counter()
        - inicio
    )

    from Solver.ray_tracing import RayTracingResult

    resultado = RayTracingResult(
        estado=estado,
        status=status,
        parametro=parametro,
        pasos_aceptados=pasos_aceptados,
        pasos_rechazados=pasos_rechazados,
    )

    return (
        resultado,
        elapsed,
        bloques_procesados,
    )


# =============================================================================
# COMPARACIÓN DE RESULTADOS
# =============================================================================

def comparar_resultados(
    referencia,
    prueba,
):

    status_diferentes = np.count_nonzero(
        referencia.status
        != prueba.status
    )

    pasos_diff = np.max(
        np.abs(
            referencia.pasos_aceptados
            - prueba.pasos_aceptados
        )
    )

    rechazos_diff = np.max(
        np.abs(
            referencia.pasos_rechazados
            - prueba.pasos_rechazados
        )
    )

    estado_error = np.max(
        np.abs(
            referencia.estado
            - prueba.estado
        )
    )

    parametro_error = np.max(
        np.abs(
            referencia.parametro
            - prueba.parametro
        )
    )

    return (
        status_diferentes,
        pasos_diff,
        rechazos_diff,
        estado_error,
        parametro_error,
    )


# =============================================================================
# ESTADOS
# =============================================================================

def contar_estados(resultado):

    return {
        "escape": np.count_nonzero(
            resultado.status
            == STATUS_ESCAPE
        ),

        "horizon": np.count_nonzero(
            resultado.status
            == STATUS_HORIZON
        ),

        "max_lambda": np.count_nonzero(
            resultado.status
            == STATUS_MAX_LAMBDA
        ),

        "step_failure": np.count_nonzero(
            resultado.status
            == STATUS_STEP_FAILURE
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 90)
    print(
        "SCHWARZSCHILD — BENCHMARK DE TAMAÑO DE BATCH"
    )
    print("=" * 90)

    print(
        f"resolución        = {RESOLUCION}"
    )

    print(
        f"rayos             = "
        f"{SIZE * SIZE:,}"
    )

    print(
        f"FOV               = "
        f"{FOV:.2f}°"
    )

    print(
        f"R0                = "
        f"{R0:.6f}"
    )

    print(
        f"rtol / atol       = "
        f"{RTOL:.1e} / {ATOL:.1e}"
    )

    print(
        f"h0 / h_max       = "
        f"{H0:.3e} / {H_MAX:.3e}"
    )

    metrica = Schwarzschild(
        M=M
    )

    x0, tetrada = construir_observador(
        metrica
    )

    _, y0 = construir_rayos(
        x0,
        tetrada,
    )

    eventos = construir_eventos()

    diagnostico_camara(
        metrica,
        y0,
    )

    # Un único solver para todas las pruebas.
    # Así el GeodesicaEvaluator se construye una sola vez.
    solver = SolverRayTracing(
        metrica,
        rtol=RTOL,
        atol=ATOL,
    )

    # -------------------------------------------------------------------------
    # REFERENCIA
    # -------------------------------------------------------------------------

    print()
    print("=" * 90)
    print("REFERENCIA — BATCH COMPLETO")
    print("=" * 90)

    referencia, tiempo_ref = resolver_completo(
        solver,
        y0,
        eventos,
    )

    counts = contar_estados(
        referencia
    )

    print(
        f"tiempo             = "
        f"{tiempo_ref:.6f} s"
    )

    print(
        f"escape             = "
        f"{counts['escape']:,}"
    )

    print(
        f"horizon            = "
        f"{counts['horizon']:,}"
    )

    print(
        f"max_lambda         = "
        f"{counts['max_lambda']:,}"
    )

    print(
        f"step_failure       = "
        f"{counts['step_failure']:,}"
    )

    print(
        f"pasos aceptados    = "
        f"{referencia.pasos_aceptados.sum():,}"
    )

    print(
        f"rechazos           = "
        f"{referencia.pasos_rechazados.sum():,}"
    )

    # -------------------------------------------------------------------------
    # SWEEP
    # -------------------------------------------------------------------------

    print()
    print("=" * 90)
    print("SWEEP DE TAMAÑO DE BLOQUE")
    print("=" * 90)

    print(
        f"{'bloque':>10s} "
        f"{'n bloques':>12s} "
        f"{'tiempo [s]':>14s} "
        f"{'speedup':>10s} "
        f"{'escape':>10s} "
        f"{'horizon':>10s} "
        f"{'Δ estado':>10s} "
        f"{'Δ y máx':>14s}"
    )

    print(
        "-" * 90
    )

    resultados = []

    # Para evitar introducir sesgo por el orden de las pruebas,
    # hacemos primero algunos warmups baratos con bloques pequeños.
    for bloque in BLOQUES:

        if bloque is None:

            resultado = referencia
            elapsed = tiempo_ref
            n_bloques = 1

        else:

            (
                resultado,
                elapsed,
                n_bloques,
            ) = resolver_bloques(
                solver,
                y0,
                eventos,
                bloque,
            )

        (
            status_diff,
            pasos_diff,
            rechazos_diff,
            estado_error,
            parametro_error,
        ) = comparar_resultados(
            referencia,
            resultado,
        )

        speedup = (
            tiempo_ref / elapsed
        )

        print(
            f"{str(bloque):>10s} "
            f"{n_bloques:12d} "
            f"{elapsed:14.6f} "
            f"{speedup:10.4f} "
            f"{np.count_nonzero(resultado.status == STATUS_ESCAPE):10,d} "
            f"{np.count_nonzero(resultado.status == STATUS_HORIZON):10,d} "
            f"{status_diff:10,d} "
            f"{estado_error:14.3e}"
        )

        resultados.append(
            {
                "bloque": bloque,
                "tiempo": elapsed,
                "speedup": speedup,
                "n_bloques": n_bloques,
                "status_diff": status_diff,
                "pasos_diff": pasos_diff,
                "rechazos_diff": rechazos_diff,
                "estado_error": estado_error,
                "parametro_error": parametro_error,
                "resultado": resultado,
            }
        )

    # -------------------------------------------------------------------------
    # ANÁLISIS
    # -------------------------------------------------------------------------

    candidatos = [
        r
        for r in resultados
        if r["status_diff"] == 0
    ]

    mejor = min(
        candidatos,
        key=lambda r: r["tiempo"],
    )

    print()
    print("=" * 90)
    print("RESULTADO")
    print("=" * 90)

    print(
        f"mejor bloque       = "
        f"{mejor['bloque']}"
    )

    print(
        f"tiempo             = "
        f"{mejor['tiempo']:.6f} s"
    )

    print(
        f"speedup            = "
        f"{mejor['speedup']:.4f}x"
    )

    print(
        f"n bloques          = "
        f"{mejor['n_bloques']}"
    )

    print()

    if mejor["speedup"] > 1.10:

        print(
            "→ Hay una mejora de al menos 10% "
            "al limitar el tamaño del batch."
        )

        print(
            "  Vale la pena incorporar un "
            "batch máximo configurable al solver."
        )

    elif mejor["speedup"] > 1.03:

        print(
            "→ Hay una mejora pequeña pero "
            "medible."
        )

        print(
            "  Conviene evaluar el coste de "
            "mantener bloques fijos."
        )

    elif mejor["speedup"] > 0.98:

        print(
            "→ El rendimiento es prácticamente "
            "equivalente."
        )

        print(
            "  El batch completo ya está cerca "
            "del óptimo."
        )

    else:

        print(
            "→ Separar el batch empeora el "
            "rendimiento."
        )

        print(
            "  Mantener el batch completo."
        )

    print()
    print("=" * 90)
    print("FIN DEL BENCHMARK")
    print("=" * 90)


if __name__ == "__main__":
    main()