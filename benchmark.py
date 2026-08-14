import time
import numpy as np

from Metricas.schwarzschild import Schwarzschild
from Solver.geodesica import GeodesicaEvaluator
from Solver.solver import SolverGeodesica


# ============================================================
# CONFIGURACIÓN
# ============================================================

N = 100_000
REPETICIONES = 7

M = 1

X0 = np.array(
    [
        0.0,
        10.0,
        np.pi / 2,
        0.0,
    ],
    dtype=float,
)

U0 = np.array(
    [
        1.2,
        -0.05,
        0.02,
        0.08,
    ],
    dtype=float,
)

INTERVALO = (
    0.0,
    100.0,
)

RTOL = 1e-12
ATOL = 1e-14


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def medir(func, n=N, repeticiones=REPETICIONES):

    tiempos = []

    for _ in range(repeticiones):

        t0 = time.perf_counter()

        for _ in range(n):
            func()

        tiempos.append(
            time.perf_counter() - t0
        )

    tiempos = np.asarray(
        tiempos,
        dtype=float,
    )

    mediana = np.median(tiempos)

    return (
        tiempos,
        mediana,
        n / mediana,
    )


def imprimir_medicion(tiempos, mediana, rate):

    for i, tiempo in enumerate(
        tiempos,
        start=1,
    ):
        print(
            f"  run {i}: "
            f"{tiempo:.6f} s"
        )

    print()
    print(
        f"  mediana: "
        f"{mediana:.6f} s"
    )

    print(
        f"  rate:    "
        f"{rate:,.0f} eval/s"
    )

    print(
        f"  variación: "
        f"{np.std(tiempos) / mediana * 100:.2f}%"
    )


# ============================================================
# CONSTRUCCIÓN
# ============================================================

print("=" * 60)
print("CONSTRUCCIÓN")
print("=" * 60)

metric = Schwarzschild(M=M)

t0 = time.perf_counter()

evaluator = GeodesicaEvaluator(
    metric
)

t_evaluator = time.perf_counter() - t0


t0 = time.perf_counter()

solver = SolverGeodesica(
    metric
)

t_solver = time.perf_counter() - t0


print()
print(
    f"GeodesicaEvaluator : "
    f"{t_evaluator:.6f} s"
)

print(
    f"SolverGeodesica    : "
    f"{t_solver:.6f} s"
)


# ============================================================
# ESTADO
# ============================================================

y = np.empty(
    2 * evaluator.dim,
    dtype=float,
)

y[:evaluator.dim] = X0
y[evaluator.dim:] = U0


out = np.empty_like(y)


# ============================================================
# VERIFICACIÓN DE RHS
# ============================================================

print()
print("=" * 60)
print("VERIFICACIÓN")
print("=" * 60)

evaluator.sistema(
    y,
    out,
)

rhs_solver = solver._rhs(
    0.0,
    y,
)

print()
print("evaluator.sistema():")
print(out)

print()
print("solver._rhs():")
print(rhs_solver)

if not np.allclose(
    out,
    rhs_solver,
    rtol=1e-13,
    atol=1e-15,
):

    raise RuntimeError(
        "evaluator.sistema() y "
        "solver._rhs() no coinciden"
    )

print()
print("✓ RHS coinciden")


# ============================================================
# WARM-UP
# ============================================================

for _ in range(10_000):

    evaluator.sistema(
        y,
        out,
    )

    solver._rhs(
        0.0,
        y,
    )


# ============================================================
# BENCHMARK 1
# EVALUATOR
# ============================================================

print()
print("=" * 60)
print("BENCHMARK EVALUATOR")
print("=" * 60)

tiempos_evaluator, mediana_evaluator, rate_evaluator = medir(
    lambda: evaluator.sistema(
        y,
        out,
    )
)

print()

imprimir_medicion(
    tiempos_evaluator,
    mediana_evaluator,
    rate_evaluator,
)


# ============================================================
# BENCHMARK 2
# SOLVER._rhs
# ============================================================

print()
print("=" * 60)
print("BENCHMARK SOLVER._rhs")
print("=" * 60)

tiempos_rhs, mediana_rhs, rate_rhs = medir(
    lambda: solver._rhs(
        0.0,
        y,
    )
)

print()

imprimir_medicion(
    tiempos_rhs,
    mediana_rhs,
    rate_rhs,
)


# ============================================================
# COMPARACIÓN EVALUATOR → RHS
# ============================================================

print()
print("=" * 60)
print("OVERHEAD DEL CALLBACK")
print("=" * 60)

print(
    f"  evaluator.sistema : "
    f"{rate_evaluator:,.0f} eval/s"
)

print(
    f"  solver._rhs       : "
    f"{rate_rhs:,.0f} eval/s"
)

print()

print(
    f"  overhead relativo: "
    f"{rate_evaluator / rate_rhs:.2f}x"
)


# ============================================================
# BENCHMARK 3
# SOLVE_IVP
# ============================================================

print()
print("=" * 60)
print("BENCHMARK SOLVE_IVP")
print("=" * 60)

# Warm-up de integración
solver.resolver(
    X0,
    U0,
    INTERVALO,
    metodo="DOP853",
    rtol=RTOL,
    atol=ATOL,
)


tiempos_ivp = []
resultados = []

for _ in range(REPETICIONES):

    t0 = time.perf_counter()

    resultado = solver.resolver(
        X0,
        U0,
        INTERVALO,
        metodo="DOP853",
        rtol=RTOL,
        atol=ATOL,
    )

    elapsed = time.perf_counter() - t0

    tiempos_ivp.append(
        elapsed
    )

    resultados.append(
        resultado
    )


tiempos_ivp = np.asarray(
    tiempos_ivp,
    dtype=float,
)

mediana_ivp = np.median(
    tiempos_ivp
)

resultado = resultados[
    np.argmin(
        np.abs(
            tiempos_ivp
            - mediana_ivp
        )
    )
]

print()

for i, tiempo in enumerate(
    tiempos_ivp,
    start=1,
):

    print(
        f"  run {i}: "
        f"{tiempo:.6f} s"
    )

print()

print(
    f"  mediana: "
    f"{mediana_ivp:.6f} s"
)

print(
    f"  variación: "
    f"{np.std(tiempos_ivp) / mediana_ivp * 100:.2f}%"
)

print()

print(
    f"  éxito:       "
    f"{resultado.success}"
)

print(
    f"  estado:      "
    f"{resultado.status}"
)

print(
    f"  evaluaciones: "
    f"{resultado.nfev:,}"
)

print(
    f"  pasos:        "
    f"{len(resultado.t):,}"
)

print(
    f"  lambda final: "
    f"{resultado.t[-1]:.8f}"
)

print()

print(
    f"  RHS/s:        "
    f"{resultado.nfev / mediana_ivp:,.0f}"
)

# ============================================================
# BENCHMARK 530 CALLS
# ============================================================

print()
print("=" * 60)
print("BENCHMARK 530 CALLS _rhs")
print("=" * 60)

N_CALLS = resultado.nfev

tiempos_calls = []

for _ in range(REPETICIONES):

    t0 = time.perf_counter()

    for _ in range(N_CALLS):

        solver._rhs(
            0.0,
            y,
        )

    elapsed = time.perf_counter() - t0

    tiempos_calls.append(
        elapsed
    )

tiempos_calls = np.asarray(
    tiempos_calls,
    dtype=float,
)

mediana_calls = np.median(
    tiempos_calls
)

print()

for i, tiempo in enumerate(
    tiempos_calls,
    start=1,
):

    print(
        f"  run {i}: "
        f"{tiempo:.8f} s"
    )

print()

print(
    f"  mediana: "
    f"{mediana_calls:.8f} s"
)

print(
    f"  tiempo/RHS: "
    f"{mediana_calls / N_CALLS * 1e6:.2f} µs"
)

print(
    f"  RHS/s: "
    f"{N_CALLS / mediana_calls:,.0f}"
)

print()

print(
    f"  solve_ivp: "
    f"{mediana_ivp:.8f} s"
)

print(
    f"  overhead solve_ivp: "
    f"{mediana_ivp - mediana_calls:.8f} s"
)

print(
    f"  factor: "
    f"{mediana_ivp / mediana_calls:.2f}x"
)

print("=" * 60)


# ============================================================
# COMPARACIÓN FINAL
# ============================================================

print()
print("=" * 60)
print("COMPARACIÓN FINAL")
print("=" * 60)

print()
print(
    f"Evaluator.sistema()"
)

print(
    f"  {rate_evaluator:,.0f} eval/s"
)

print()
print(
    f"Solver._rhs()"
)

print(
    f"  {rate_rhs:,.0f} eval/s"
)

print()
print(
    f"solve_ivp"
)

print(
    f"  {resultado.nfev / mediana_ivp:,.0f} RHS/s"
)

print()

print(
    "Relaciones:"
)

print(
    f"  evaluator / _rhs : "
    f"{rate_evaluator / rate_rhs:.2f}x"
)

print(
    f"  _rhs / solve_ivp : "
    f"{rate_rhs / (resultado.nfev / mediana_ivp):.2f}x"
)

print(
    f"  evaluator / solve_ivp : "
    f"{rate_evaluator / (resultado.nfev / mediana_ivp):.2f}x"
)

print()
print("=" * 60)