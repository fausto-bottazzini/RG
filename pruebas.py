import time
import numpy as np

from Metricas.kerr import Kerr
from Solver.solver import SolverGeodesica


# ============================================================
# CONFIGURACIÓN
# ============================================================

M = 1.0
a = 0.5

x0 = np.array([
    0.0,          # t
    10.0,         # r
    np.pi / 2,    # theta
    0.0,          # phi
])

u0 = np.array([
    1.2,          # dt/dlambda
    -0.05,        # dr/dlambda
    0.02,         # dtheta/dlambda
    0.08,         # dphi/dlambda
])

intervalo = (
    0.0,
    100.0,
)


# ============================================================
# MÉTRICA
# ============================================================

print("=" * 60)
print("PRUEBA KERR")
print("=" * 60)

metric = Kerr(
    M=M,
    a=a,
)

print()
print("Métrica construida")
print(f"  M = {M}")
print(f"  a = {a}")
print(f"  dimensión = {metric.g.dim}")


# ============================================================
# EVALUADOR
# ============================================================

print()
print("=" * 60)
print("GEODESICA EVALUATOR")
print("=" * 60)

t0 = time.perf_counter()

solver = SolverGeodesica(metric)

tiempo = time.perf_counter() - t0

print(
    f"  construcción: {tiempo:.6f} s"
)


# ============================================================
# RHS INICIAL
# ============================================================

y0 = np.concatenate([
    x0,
    u0,
])

rhs = solver._rhs(
    0.0,
    y0,
)

print()
print("RHS inicial:")
print(rhs)


# ============================================================
# INTEGRACIÓN
# ============================================================

print()
print("=" * 60)
print("SOLVER")
print("=" * 60)

t0 = time.perf_counter()

resultado = solver.resolver(
    x0,
    u0,
    intervalo,
    metodo="DOP853",
    rtol=1e-9,
    atol=1e-11,
)

tiempo = time.perf_counter() - t0


# ============================================================
# RESULTADOS
# ============================================================

print()
print("Resultado")

print(
    f"  éxito       : {resultado.success}"
)

print(
    f"  estado      : {resultado.status}"
)

print(
    f"  mensaje     : {resultado.message}"
)

print(
    f"  tiempo      : {tiempo:.6f} s"
)

print(
    f"  evaluaciones: {resultado.nfev}"
)

print(
    f"  pasos       : {len(resultado.t)}"
)

print(
    f"  lambda final: {resultado.t[-1]:.8f}"
)

if tiempo > 0:

    print(
        f"  RHS/s       : "
        f"{resultado.nfev / tiempo:,.0f}"
    )


# ============================================================
# ESTADOS
# ============================================================

print()
print("Estado inicial:")
print(y0)

print()
print("Estado final:")
print(resultado.y[:, -1])


# ============================================================
# FIN
# ============================================================

print()
print("=" * 60)