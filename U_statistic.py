from V_statistic import V_statistic, V_stats_torch
from U2V import partition, encode_partition, partition_weights
import numpy as np


def U_stats(Tensor_list: list[np.ndarray], order: int):
    n = Tensor_list[0].shape[0]
    partitions = []
    for k in range(1, order + 1):
        partitions.extend(partition(order, k))
    weights = []
    for partition_ in partitions:
        weights.append(partition_weights(partition_))
    return np.sum(
        [
            V_statistic(Tensor_list, V_sequence=encode_partition(partition_)) * weight
            for partition_, weight in zip(partitions, weights)
        ]
    ) / np.prod(np.arange(n, n - order, -1))


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
