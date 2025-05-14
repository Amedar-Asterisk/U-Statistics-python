import opt_einsum as oe
import numpy as np


def full_bipart(set1, set2):
    """
    Generate a full bipartite graph between two sets of indices.
    """
    result = ""
    for i in set1:
        for j in set2:
            result += f"{i}{j},"
    return result[:-1]


n = 2
m = 9
A = np.random.rand(n, n)
T = [A] * m
set1 = "abc"
set2 = "efg"
format = full_bipart(set1, set2) + "->"
print(format)
path1 = oe.contract_path(format, *T, optimize="greedy")
path2 = oe.contract_path(format, *T, optimize="optimal")
print(path1[1])
print()
print(path2[1])
