import time
import sympy as sp
from Metricas.schwarzschild import Schwarzschild
from Metricas.kerr import Kerr

def bench(nombre, f):
    t0 = time.perf_counter()
    obj = f()
    print(f"{nombre:<24} {time.perf_counter()-t0:8.5f} s")
    return obj

def nnz(T):
    return getattr(T, "nnz", "-")

def ops(expr):
    if hasattr(expr, "items"):
        return sum(sp.count_ops(v) for _, v in expr.items())
    return sp.count_ops(expr)

def chars(expr):
    if hasattr(expr, "items"):
        return sum(len(str(v)) for _, v in expr.items())
    return len(str(expr))

def compare(expr, nombre):
    methods = {
        "factor": sp.factor,
        "cancel": sp.cancel,
        "together": sp.together,
        "simplify": sp.simplify,
    }

    print(f"\n{nombre}")
    print(f"{'Original':<10} ops={ops(expr):5}")

    for m, fun in methods.items():
        t0 = time.perf_counter()
        out = fun(expr)
        dt = time.perf_counter() - t0
        print(f"{m:<10} {dt:8.5f} s  ops={ops(out):5}")

M = sp.symbols("M")
a = sp.symbols("a")
metric = bench("Crear métrica", lambda: Kerr(M, a))

g       = bench("Métrica", lambda: metric.g)
g_inv   = bench("Inversa", lambda: metric.g_inv)
dg      = bench("∂g", lambda: metric.derivadas())
Gamma   = bench("Christoffel", lambda: metric.christoffel())
dGamma  = bench("∂Γ", lambda: metric.derivadas_christoffel())
Riemann = bench("Riemann", lambda: metric.riemann())
Ricci   = bench("Ricci", lambda: metric.ricci())
R       = bench("Escalar", lambda: metric.escalar_curvatura())

print("\nTensor                 nnz     ops")
print("--------------------------------------")
for nombre, T in [
    ("g", g),
    ("g⁻¹", g_inv),
    ("∂g", dg),
    ("Γ", Gamma),
    ("∂Γ", dGamma),
    ("Riemann", Riemann),
    ("Ricci", Ricci),
]:
    print(f"{nombre:<12} {nnz(T):>4} {ops(T):>8}")

compare(R, "Escalar de curvatura")

for (idx, val) in Ricci.items():
    compare(val, f"Ricci{idx}")