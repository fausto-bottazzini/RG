from Tensores.tensor import Tensor


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