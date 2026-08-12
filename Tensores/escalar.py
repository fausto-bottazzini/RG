import sympy as sp

class EscalarCurvatura:
    """Calcula el escalar de curvatura a partir del tensor de Ricci"""

    def __init__(self, metrica):
        self.metrica = metrica
        self.value = self.compute()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("metrica", None)
        return state

    def compute(self):
        g_inv = self.metrica.g_inv
        Ricci = self.metrica.ricci()

        value = sp.Integer(0)

        for (mu,nu), gij in g_inv.items():

            value += gij * Ricci[mu,nu]

        return value