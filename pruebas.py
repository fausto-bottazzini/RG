from Metricas.schwarzschild import Schwarzschild

g = Schwarzschild(1)

Gamma = g.numeric("christoffel")

print(Gamma)
print(Gamma.indices)
print(Gamma.evaluar(0.0, 10.0, 1.0, 0.5))
print(Gamma.evaluar_valores(0.0, 10.0, 1.0, 0.5))