import time
import numpy as np

from Metricas.schwarzschild import Schwarzschild
from Solver.solver import SolverGeodesica
from Solver.invariantes import Invariantes


# ============================================================
# CONFIGURACIÓN
# ============================================================

M = 1.0

RTOLS = [
    1e-6,
    1e-7,
    1e-8,
    1e-9,
    1e-10,
]

ATOL_FACTOR = 1e-2

METODO = "DOP853"

INTERVALO = (0.0, 5000.0)


# ============================================================
# CASOS
# ============================================================

CASOS = {

    "orbita circular": {

        "x0": np.array([
            0.0,
            10.0,
            np.pi / 2,
            0.0,
        ]),

        "u0": np.array([
            1.19522861,
            0.0,
            0.0,
            0.03779645,
        ]),
    },


    "orbita no circular": {

        "x0": np.array([
            0.0,
            10.0,
            np.pi / 2,
            0.0,
        ]),

        "u0": np.array([
            1.2,
            -0.05,
            0.02,
            0.08,
        ]),
    },


    "fuerte campo": {

        "x0": np.array([
            0.0,
            5.0,
            np.pi / 2,
            0.0,
        ]),

        "u0": np.array([
            1.1,
            -0.05,
            0.01,
            0.15,
        ]),
    },
}


# ============================================================
# MÉTRICA
# ============================================================

metric = Schwarzschild(M=M)

solver = SolverGeodesica(metric)

inv = Invariantes(metric)


# ============================================================
# REFERENCIA
# ============================================================

def construir_referencia(x0, u0):

    return solver.resolver(
        x0,
        u0,
        INTERVALO,
        metodo=METODO,
        rtol=1e-13,
        atol=1e-15,
    )


# ============================================================
# ERROR DE ESTADO
# ============================================================

def error_estado(resultado, referencia):

    y = resultado.y[:, -1]
    y_ref = referencia.y[:, -1]

    escala = np.maximum(
        np.abs(y_ref),
        1.0,
    )

    return np.max(
        np.abs(y - y_ref) / escala
    )


# ============================================================
# EJECUCIÓN DE UN CASO
# ============================================================

def benchmark_caso(nombre, x0, u0):

    print()
    print("=" * 80)
    print(nombre.upper())
    print("=" * 80)

    # --------------------------------------------------------
    # Referencia
    # --------------------------------------------------------

    print()
    print("REFERENCIA")

    t0 = time.perf_counter()

    referencia = construir_referencia(
        x0,
        u0,
    )

    tiempo_ref = time.perf_counter() - t0

    print(
        f"  tiempo      : {tiempo_ref:.6f} s"
    )

    print(
        f"  nfev        : {referencia.nfev}"
    )

    print(
        f"  pasos       : {len(referencia.t)}"
    )

    # --------------------------------------------------------
    # Cabecera
    # --------------------------------------------------------

    print()
    print(
        f"{'rtol':>10}"
        f"{'tiempo':>12}"
        f"{'nfev':>8}"
        f"{'pasos':>8}"
        f"{'err estado':>15}"
        f"{'Δnorma':>15}"
        f"{'ΔE':>15}"
        f"{'ΔL':>15}"
    )

    print("-" * 80)

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    for rtol in RTOLS:

        atol = rtol * ATOL_FACTOR

        t0 = time.perf_counter()

        resultado = solver.resolver(
            x0,
            u0,
            INTERVALO,
            metodo=METODO,
            rtol=rtol,
            atol=atol,
        )

        tiempo = time.perf_counter() - t0

        # --------------------------------------------
        # Error contra referencia
        # --------------------------------------------

        err_y = error_estado(
            resultado,
            referencia,
        )

        # --------------------------------------------
        # Conservación
        # --------------------------------------------

        errores = inv.errores(
            resultado
        )

        d_norma = errores[
            "norma"
        ]["max_absoluto"]

        d_energia = errores[
            "energia"
        ]["max_absoluto"]

        d_momento = errores[
            "momento_angular"
        ]["max_absoluto"]

        # --------------------------------------------
        # Resultado
        # --------------------------------------------

        print(
            f"{rtol:10.1e}"
            f"{tiempo:12.6f}"
            f"{resultado.nfev:8d}"
            f"{len(resultado.t):8d}"
            f"{err_y:15.3e}"
            f"{d_norma:15.3e}"
            f"{d_energia:15.3e}"
            f"{d_momento:15.3e}"
        )


# ============================================================
# EJECUTAR TODOS LOS CASOS
# ============================================================

print()
print("=" * 80)
print("BENCHMARK DE REGÍMENES — SCHWARZSCHILD")
print("=" * 80)

print()
print(f"Método : {METODO}")
print(f"M      : {M}")
print(f"λ      : {INTERVALO}")
print(f"Casos  : {len(CASOS)}")


for nombre, datos in CASOS.items():

    benchmark_caso(
        nombre,
        datos["x0"],
        datos["u0"],
    )


# ============================================================
# FIN
# ============================================================

print()
print("=" * 80)
print("FIN DEL BENCHMARK")
print("=" * 80)