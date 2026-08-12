import sympy as sp

class CacheNumerico:
    def __init__(self, metric):
        self.metric = metric
        self.memory = {}

    def get(self, name, tensor, modules="numpy"):
        if name in self.memory:
            return self.memory[name]

        funciones = {}
        for idx, expr in tensor.items():
            funciones[idx] = sp.lambdify(self.metric.coords, expr, modules=modules, cse=True)

        self.memory[name] = funciones
        return funciones

    def clear(self):
        self.memory.clear()
    