import numpy as np

def construir_geodesica(solver, condiciones, r0, vr, vphi, *, tipo, lambda_max):
    x0 = np.array([0.0, r0, np.pi / 2, 0.0], dtype=float)
    u0 = condiciones.normalizar(x0, np.array([1.0, vr, 0.0, vphi], dtype=float), tipo=tipo, componente=0, signo=+1)
    return solver.resolver(x0, u0, (0.0, lambda_max), metodo="DOP853", rtol=1e-9, atol=1e-11)