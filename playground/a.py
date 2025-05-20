from typing import Dict, List, Tuple, Union
import itertools
import numpy as np


def _mask_tensors(
    tensors: Dict[int, np.ndarray], sample_size: int
) -> Dict[int, np.ndarray]:
    shapes = {index: tensor.ndim for index, tensor in tensors.items()}
    for index, ndim in shapes.items():
        if ndim > 1:
            mask_total = np.zeros((sample_size,) * ndim, dtype=bool)
            for i, j in itertools.combinations(range(ndim), 2):
                mask = _mask_tensor(ndim, sample_size, i, j)
                mask_total |= mask
            mask_total = np.logical_not(mask_total)
            tensors[index] = tensors[index][mask_total]
    return tensors


def _mask_tensor(ndim: int, dim: int, index1: int, index2: int) -> np.ndarray:
    shape1 = [1] * ndim
    shape1[index1] = dim
    shape2 = [1] * ndim
    shape2[index2] = dim

    idx1 = np.arange(dim).reshape(shape1)
    idx2 = np.arange(dim).reshape(shape2)
    mask = idx1 == idx2
    mask = np.broadcast_to(mask, (dim,) * ndim)
    return mask


mask = _mask_tensor(2, 4, 0, 1)
print(mask)
mask = np.logical_not(mask)
print(mask)

A = np.random.randn(4, 4)
print(A[mask])
