import numpy as np

from Metricas.schwarzschild import Schwarzschild
from Solver.ray_tracing import SolverRayTracing
from Tensores.operaciones import tetrada_obs, transformar_vector
from Visualizador.RayTracing.cam import Camara
from Solver.solver import SolverGeodesica

def matriz_metrica(metrica, x):
    metric_num = metrica.numeric("metric")
    valores = metric_num.evaluar_valores(*x)

    g = np.zeros(
        (metrica.g.dim, metrica.g.dim),
        dtype=float,
    )

    for idx, valor in zip(
        metric_num.indices,
        valores,
    ):
        g[idx] = valor

    return g

# ============================================================
# CONFIGURACIÓN
# ============================================================

metrica = Schwarzschild(M=1.0)

x0 = np.array([
    0.0,
    10.0,
    np.pi / 2,
    0.0,
])


# ============================================================
# MÉTRICA
# ============================================================

metric_num = metrica.numeric("metric")

valores = metric_num.evaluar_valores(*x0)

g = np.zeros((4, 4), dtype=float)

for idx, valor in zip(metric_num.indices, valores):
    g[idx] = valor


# ============================================================
# OBSERVADOR
# ============================================================

u0 = np.zeros(4)
u0[0] = 1.0 / np.sqrt(-g[0, 0])

print("=" * 70)
print("TEST RAY TRACING")
print("=" * 70)

print("\nNorma observador:")
print(u0 @ g @ u0)


# ============================================================
# TÉTRADA
# ============================================================

e = tetrada_obs(
    metrica,
    x0,
    u0,
)

print("\nTetrada:")
print(e)

ortonormalidad = np.einsum(
    "ai,ij,bj->ab",
    e,
    g,
    e,
)

print("\nOrtonormalidad:")
print(ortonormalidad)


# ============================================================
# CÁMARA
# ============================================================

camara = Camara(
    posicion=x0,
    resolucion=(10, 10),
    fov=np.radians(20.0),
)

directions = camara.rays()
k_local = camara.rays_local()

k = transformar_vector(
    e,
    k_local,
)

print("\nCámara:")
print("directions =", directions.shape)
print("k_local    =", k_local.shape)
print("k          =", k.shape)


# ============================================================
# NULIDAD INICIAL
# ============================================================

normas = np.einsum(
    "ni,ij,nj->n",
    k,
    g,
    k,
)

print("\nNulidad inicial:")
print(
    "máximo |k·k| =",
    np.max(np.abs(normas)),
)


# ============================================================
# ESTADO
# ============================================================

y0 = np.empty(
    (len(k), 8),
    dtype=float,
)

y0[:, :4] = x0
y0[:, 4:] = k

# Un solo rayo para los tests numéricos
y = y0[:1].copy()


# ============================================================
# SOLVER
# ============================================================

solver = SolverRayTracing(
    metrica,
    rtol=1e-9,
    atol=1e-11,
)


# ============================================================
# TEST 1 — RHS
# ============================================================

print("\n" + "=" * 70)
print("TEST 1 — RHS")
print("=" * 70)

rhs = solver.rhs(y)

print("shape =", rhs.shape)
print("finito =", np.all(np.isfinite(rhs)))


# ============================================================
# TEST 2 — UN PASO RK4
# ============================================================

h = 0.01

print("\n" + "=" * 70)
print("TEST 2 — UN PASO RK4")
print("=" * 70)

y_rk4 = solver.step_rk4(
    y,
    h,
)

print("estado inicial:")
print(y[0])

print("\nestado RK4:")
print(y_rk4[0])


# ============================================================
# TEST 3 — UN PASO RK45
# ============================================================

print("\n" + "=" * 70)
print("TEST 3 — UN PASO RK45")
print("=" * 70)

h_batch = np.array([0.01])

y_rk45, error_rk45 = solver._rk45(
    y,
    h_batch,
)

print("estado RK45:")
print(y_rk45[0])

print("\nerror estimado:")
print(error_rk45)

error_step = np.max(
    np.abs(
        y_rk45[0]
        - y_rk4[0]
    )
)

print(
    "\nerror RK45 vs RK4 =",
    error_step,
)


# ============================================================
# TEST 4 — RESOLVER
# ============================================================

print("\n" + "=" * 70)
print("TEST 4 — RESOLVER")
print("=" * 70)

resultado = solver.resolver(
    y,
    lambda_max=20.0,
    h0=0.01,
    h_min=1e-8,
    h_max=0.01,
)

yf = resultado.estado[0]

print("\nestado final:")
print(yf)

print("\nstatus:")
print(resultado.status)

print("\npasos aceptados:")
print(resultado.pasos_aceptados)

print("\npasos rechazados:")
print(resultado.pasos_rechazados)

print("\nparámetro final:")
print(resultado.parametro)


# ============================================================
# NULIDAD FINAL
# ============================================================

g_final = matriz_metrica(
    metrica,
    yf[:4],
)

kf = yf[4:]

norma_final = kf @ g_final @ kf
print("\nNulidad final:")
print(
    "k·k =",
    norma_final,
)


# ============================================================
# TEST 5 — RESOLVER PRO
# ============================================================

print("\n" + "=" * 70)
print("TEST 5 — RESOLVER PRO")
print("=" * 70)

resultado_pro = solver.resolver_pro(
    y,
    lambda_max=20.0,
    h0=0.01,
    h_min=1e-8,
    h_max=0.01,
)

yf_pro = resultado_pro.estado[0]

print("\nestado final:")
print(yf_pro)

print("\nstatus:")
print(resultado_pro.status)

print("\npasos aceptados:")
print(resultado_pro.pasos_aceptados)

print("\npasos rechazados:")
print(resultado_pro.pasos_rechazados)

print("\nparámetro final:")
print(resultado_pro.parametro)

g_final_pro = matriz_metrica(
    metrica,
    yf_pro[:4],
)

kf_pro = yf_pro[4:]

norma_pro = kf_pro @ g_final_pro @ kf_pro

print("\nNulidad final:")
print(
    "k·k =",
    norma_pro,
)


# ============================================================
# TEST 6 — RESOLVER VS RESOLVER PRO
# ============================================================

print("\n" + "=" * 70)
print("TEST 6 — RESOLVER VS RESOLVER PRO")
print("=" * 70)

error_solvers = np.max(
    np.abs(
        resultado.estado[0]
        - resultado_pro.estado[0]
    )
)

print(
    "error máximo =",
    error_solvers,
)


# ============================================================
# TEST 7 — TOLERANCIAS
# ============================================================

print("\n" + "=" * 70)
print("TEST 7 — DEPENDENCIA CON RTOL")
print("=" * 70)

for rtol, atol in [
    (1e-6, 1e-8),
    (1e-7, 1e-9),
    (1e-8, 1e-10),
    (1e-9, 1e-11),
    (1e-10, 1e-12),
]:

    solver_test = SolverRayTracing(
        metrica,
        rtol=rtol,
        atol=atol,
    )

    res = solver_test.resolver(
        y,
        lambda_max=20.0,
        h0=0.01,
        h_min=1e-10,
        h_max=0.01,
    )

    yf = res.estado[0]

    g_final = matriz_metrica(
        metrica,
        yf[:4],
    )

    kf = yf[4:]

    norma = kf @ g_final @ kf

    print(
        f"\nrtol = {rtol:.0e}"
    )

    print(
        "  pasos    =",
        res.pasos_aceptados[0],
    )

    print(
        "  rechazos =",
        res.pasos_rechazados[0],
    )

    print(
        "  |k·k|    =",
        abs(norma),
    )


solver_ref = SolverGeodesica(metrica)

resultado_ref = solver_ref.resolver(
    y[0, :4],
    y[0, 4:],
    (0.0, 20.0),
    metodo="DOP853",
    rtol=1e-10,
    atol=1e-12,
)

estado_ref = resultado_ref.y[:, -1]
estado_rt = resultado.estado[0]

error = np.max(
    np.abs(estado_rt - estado_ref)
)

print("error RayTracing vs DOP853 =", error)

print("\n" + "=" * 70)
print("FIN")
print("=" * 70)

