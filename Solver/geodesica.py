import sympy as sp

class GeodesicaEvaluator:
    """
    Evaluador numérico de la ecuación de las geodésicas.

    La ecuación completa se construye simbólicamente una sola vez
    y luego se compila mediante lambdify.
    """

    def __init__(self, metrica):
        self.metrica = metrica
        self.dim = len(metrica.coords)
        self._velocidades = sp.symbols(f"u0:{self.dim}")
        self._build_rhs()
        self._build_system()

    def _build_rhs(self):
        """
        Construye simbólicamente:
            a^mu = -Gamma^mu_ab u^a u^b
        aprovechando la simetría de Christoffel.
        """

        gamma = self.metrica.christoffel()
        u = self._velocidades
        aceleraciones = []

        for mu in range(self.dim):
            expr = sp.Integer(0)
            for (mu_i, alpha, beta), value in gamma.items():

                if mu_i != mu:
                    continue
                if alpha > beta:
                    continue

                if alpha == beta:
                    expr -= (value * u[alpha] * u[beta])
                else:
                    expr -= (2 * value * u[alpha] * u[beta])

            aceleraciones.append(expr)

        self._aceleraciones = tuple(aceleraciones)

        # sistema completo solo con coordenadas utiles

        coordinate_indices = []
        velocity_indices = []
    
        for i, coord in enumerate(self.metrica.coords):
            if any(expr.has(coord) for expr in self._aceleraciones):
                coordinate_indices.append(i)

        for i, velocity in enumerate(self._velocidades):
            if any(expr.has(velocity) for expr in self._aceleraciones):
                velocity_indices.append(i)

        self._coordinate_indices = tuple(coordinate_indices)
        self._velocity_indices = tuple(velocity_indices)

        variables = (tuple(self.metrica.coords[i] for i in self._coordinate_indices)
            + tuple(self._velocidades[i] for i in self._velocity_indices))

        self._variables = variables

        rhs = tuple(self._velocidades) + self._aceleraciones
        self._rhs_function = sp.lambdify(variables, rhs, modules="numpy", cse=True)

        self._acceleration_function = sp.lambdify(variables, self._aceleraciones, modules="numpy", cse=True)    

    def _build_system(self):
        dim = self.dim
        coordinate_indices = (self._coordinate_indices)
        velocity_indices = (self._velocity_indices)
        acceleration_function = (self._acceleration_function)

        argument_indices = (tuple(coordinate_indices) + tuple(dim + i for i in velocity_indices))
        self._argument_indices = argument_indices

        def system(y, out):
            values = acceleration_function(*(y[i] for i in argument_indices)) 
            out[:dim] = y[dim:]
            out[dim:] = values
            
        self._system = system

    def rhs(self, x, u, out):
        """
        Evalúa el sistema completo:
            [dx/dλ, du/dλ]

        x : coordenadas
        u : velocidades
        out : array de salida
        """

        values = self._rhs_function(*(x[i] for i in self._coordinate_indices),
                                    *(u[i] for i in self._velocity_indices))
        out[:] = values

    def aceleracion(self, x, u, out):
        """Evalua solamente la aceleración."""
        values = self._acceleration_function(*(x[i] for i in self._coordinate_indices),
                                             *(u[i]  for i in self._velocity_indices))
        out[:] = values

    def sistema(self, y, out):
        """Evalúa directamente el sistema completo."""
        self._system(y, out)

    @property
    def aceleraciones(self):
        return self._aceleraciones

    @property
    def rhs_simbolica(self):
        return tuple(self._velocidades) + self._aceleraciones

    @property
    def coordenadas_utilizadas(self):
        return tuple(self.metrica.coords[i] for i in self._coordinate_indices)

    @property 
    def velocidades_utilizadas(self):
        return tuple(self._velocidades[i] for i in self._velocity_indices)