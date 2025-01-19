from V_statistics import V_statistics, V_stats_torch
from U2V import partitions, encode_partition, partition_weights
import numpy as np


def U_stats(Tensor_list: list[np.ndarray], order: int):
    n_samples = Tensor_list[0].shape[0]
    for tensor in Tensor_list:
        for n in tensor.shape:
            if n != n_samples:
                raise ValueError("All indexes must have the same range of samples.")
    if order != len(Tensor_list) + Tensor_list[0].ndim - 1:
        raise ValueError("The order must match the number of tensors.")
    U_stats = 0
    for k in range(1, order + 1):
        for partition in partitions(order, k):
            U_stats += V_statistics(
                Tensor_list, V_sequence=encode_partition(partition)
            ) * partition_weights(partition)
    return U_stats / np.prod(np.arange(n_samples, n_samples - order, -1))


def U_stats_loop(tensors: list[np.ndarray], order: int):
    import itertools

    ndim = tensors[0].ndim
    nt = len(tensors)
    ns = tensors[0].shape[0]
    total_sum = 0.0
    num = 0
    for indices in itertools.permutations(range(ns), order):
        product = 1.0
        for i in range(0, nt):
            current_indices = indices[i : i + ndim]
            product *= tensors[i][current_indices]
        total_sum += product
        num += 1
    return total_sum / num
