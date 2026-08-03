import sympy as sp
from Tensores.tensor import Tensor

def simplify_tensor(tensor):
    result = Tensor(rank=tensor.rank, dim=tensor.dim)
    for idx, value in tensor.items():
        simplified = sp.simplify(value)
        if simplified != 0:
            result[idx] = simplified
    return result

def factor_tensor(tensor):
    result = Tensor(rank=tensor.rank, dim=tensor.dim)
    for idx, value in tensor.items():
        factored = sp.factor(value)
        if factored != 0:
            result[idx] = factored
    return result

def together_tensor(tensor):
    result = Tensor(rank=tensor.rank, dim=tensor.dim)
    for idx, value in tensor.items():
        togethered = sp.together(value)
        if togethered != 0:
            result[idx] = togethered
    return result