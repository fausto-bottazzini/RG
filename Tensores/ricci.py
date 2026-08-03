from Tensores.tensor import Tensor

class Ricci(Tensor):
    """Tensor de Ricci, contraccion del tensor de Riemann"""

    def __init__(self, metrica):
        super().__init__(rank=2, dim=len(metrica.coords))
        self.metrica = metrica
        self.compute()

    def compute(self):
        Riemann = self.metrica.riemann()

        for sigma in range(self.dim):
            for nu in range(self.dim):
                value = 0
                for idx, component in Riemann.select({1: sigma, 3: nu}):
                    rho = idx[0]
                    mu = idx[2]

                    if rho == mu:
                        value += component

                if value != 0:
                    self[sigma, nu] = value
                     
    