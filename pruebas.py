import numpy as np
import matplotlib.pyplot as plt

from Metricas.kerr import Kerr
from Metricas.schwarzschild import Schwarzschild
from Solver.solver import SolverGeodesica
from Visualizador.embedding import Embedding
from Solver.invariantes import Condiciones

metric = Kerr(M=1, a=0.9)
solver = SolverGeodesica(metric)
cond = Condiciones(metric)

x0 = np.array([0.0, 10.0, np.pi / 2, 0.0])
u0 = cond.normalizar(x0, np.array([1.19, 0.0, 0.0, 0.0]), tipo="timelike", componente=0, signo=+1)
print(u0)

resultado = solver.resolver(x0, u0, (0.0, 100.0), metodo="DOP853", rtol=1e-9, atol=1e-11, max_step = 0.5)
embedding = Embedding(metric, coordenadas=(1, 3), fijas={0: 0.0, 2: np.pi / 2})

r = resultado.y[1]

embedding.plot(qmin=2.005, qmax=min(100.0, r.max()*1.05), resultado=resultado, nq=100, nphi=50, cmap="gist_heat", linecolor="black")
plt.show()