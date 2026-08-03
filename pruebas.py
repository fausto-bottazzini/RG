from Metricas.kerr import Kerr
from Metricas.schwarzschild import Schwarzschild
from Visualizador.embedding import Embedding
import sympy as sp
import matplotlib.pyplot as plt

M = sp.symbols("M")
a = sp.symbols("a")

metric = Schwarzschild(1)

embedding = Embedding(metric)

fig, ax = embedding.plot(rmax=15)

plt.show()