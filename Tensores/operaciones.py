from Tensores.tensor import Tensor
import numpy as np

def suma(A, B):
    if A.rank != B.rank:
        raise ValueError("Los tensores deben tener el mismo rango para sumarse.")
    if A.dim != B.dim:
        raise ValueError("Los tensores deben tener la misma dimensión para sumarse.")
    
    C = Tensor(rank=A.rank, dim=A.dim)

    for idx, value in A.items():
        C[idx] = value
    for idx, value in B.items():
        C[idx] = C[idx] + value
    return C

def escalar(a, A):
    B = Tensor(rank=A.rank, dim=A.dim)
    for idx, value in A.items():
        B[idx] = a * value
    return B

def producto_tensorial(A, B):
    C = Tensor(rank=A.rank + B.rank, dim=A.dim)

    for idx_A, value_A in A.items():
        for idx_B, value_B in B.items():
            idx_C = idx_A + idx_B
            C[idx_C] = value_A * value_B
    return C

def contraccion(T, axis1, axis2):
    C = Tensor(rank=T.rank - 2, dim=T.dim)
    for idx, value in T.items():
        if idx[axis1] != idx[axis2]:
            continue

        new_idx = tuple(idx[i] for i in range(T.rank) if i not in (axis1, axis2))

        C[new_idx] += value
    return C

def producto_metrico(g,u,v):
    """
    Producto escalar dado por la métrica, 
        <u, v> = g_{mu nu} u^mu v^nu
    """
    return np.einsum("i,ij,j->", u, g, v)

def _proyectar_espacial(g, u, v):
    """Proyectca v sobre el subespacio ortogonal a U. (U^2 = -1)."""
    return v + producto_metrico(g, u, v) * u

def tetrada_obs(metrica, x, u, *, tolerancia=1e-12):
    """Construye la tetrada ortonormal adaptada al observador U."""
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)

    dim = metrica.g.dim

    if x.shape != (dim,): 
        raise ValueError(f"X debe tener dimensión {dim}.")
    if u.shape != (dim,):
        raise ValueError(f"U debe tener dimension {dim}.")

    metric_num = metrica.numeric("metric")
    valores = metric_num.evaluar_valores(*x)

    g = np.zeros((dim,dim), dtype=float)

    for idx, valor in zip(metric_num.indices, valores):
        g[idx] = valor
    
    norma_u = producto_metrico(g, u, u)
    if not np.isclose(norma_u, -1.0, atol=tolerancia, rtol=tolerancia):
        raise ValueError(f"El vector del observador debe estar normalizado a -1.",
                         f"Norma obtenida: {norma_u}")

    tetrada = np.empty((dim,dim), dtype=float)
    tetrada[0] = u

    candidatos = np.eye(dim)
    n_espaciales = dim - 1 
    n = 0
    for candidato in candidatos:
        v = _proyectar_espacial(g, u, candidato)

        for j in range(1, n + 1):
            e = tetrada[j]
            v -= producto_metrico(g, e, v) * e

        norma_v = producto_metrico(g, v, v)
        if norma_v <= tolerancia:
            continue

        tetrada[n + 1] = v / np.sqrt(norma_v)
        n += 1

        if n == n_espaciales:
            break
    if n != n_espaciales:
        raise ValueError("No fue posible construir una tetrada espacial completa.")

    return tetrada

def transformar_vector(e, v_loc):
    """Transforma un vector contravariante dado en una tétrada a la base coordenada."""
    e = np.asarray(e, dtype=float)
    v_loc = np.asarray(v_loc, dtype=float)
    return np.einsum("a,am ->", v_loc, e)