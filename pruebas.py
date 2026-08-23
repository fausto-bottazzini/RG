from Metricas.schwarzschild import Schwarzschild
from Solver.solver import SolverGeodesica
from Solver.invariantes import Condiciones
from Solver.ray_tracing import SolverRayTracing
from  Visualizador.RayTracing.cam import Camara
from Tensores.operaciones import transformar_vector
import numpy as np


M = 1.0
metrica = Schwarzschild(M=M)

# ------------------------------------------------------------
# Condiciones iniciales
# ------------------------------------------------------------

x0 = np.array([
    0.0,
    10.0,
    np.pi / 2,
    0.0,
])

condiciones = Condiciones(metrica)

u0 = condiciones.normalizar(
    x0,
    np.array([
        1.0,
        -0.1,
        0.0,
        0.02,
    ]),
    tipo="null",
    componente=0,
    signo=+1,
)

# ------------------------------------------------------------
# Solver tradicional
# ------------------------------------------------------------

solver = SolverGeodesica(metrica)

resultado_geo = solver.resolver(
    x0,
    u0,
    (0.0, 50.0),
    metodo="DOP853",
    rtol=1e-10,
    atol=1e-12,
)

# ------------------------------------------------------------
# Solver batch
# ------------------------------------------------------------

solver_rt = SolverRayTracing(
    metrica,
    rtol=1e-9,
    atol=1e-11,
)

camera = Camara(posicion=x0, resolucion=(20, 20), fov=np.radians(20.0))
directions = camera.rays()

k_local = camera.rays_local()

k = transformar_vector(u0, k_local)

N = len(k)

y0 = np.empty(
    (N, 8),
    dtype=float,
)

y0[:, :4] = np.asarray(
    camera.posicion,
    dtype=float,
)

y0[:, 4:] = k

resultado = solver_rt.resolver(
    y0,
    lambda_max=50.0,
    h0=0.01,
    h_min=1e-8,
    h_max=0.1,
)

print("\nRay tracing:")
print("cantidad =", N)

print("\nEstados:")
print("shape =", resultado.estado.shape)

print("\nStatus:")
unique, counts = np.unique(
    resultado.status,
    return_counts=True,
)

for s, c in zip(unique, counts):
    print(f"  {s}: {c}")

print("\nPasos:")
print(
    "mínimo =",
    resultado.pasos_aceptados.min(),
)

print(
    "máximo =",
    resultado.pasos_aceptados.max(),
)

print(
    "promedio =",
    resultado.pasos_aceptados.mean(),
)
# ------------------------------------------------------------
# Comparación
# ------------------------------------------------------------

estado_ref = resultado_geo.y[:, -1]
estado_batch = resultado.estado[0]

error = np.max(
    np.abs(estado_batch - estado_ref)
)

print("\nComparación RK4 batch vs DOP853")
print("estado final referencia:")
print(estado_ref)

print("\nestado final batch:")
print(estado_batch)

print("\nerror máximo =", error)