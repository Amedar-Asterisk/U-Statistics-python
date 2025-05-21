import numpy as np
import math


def critical_nodes(num_edges: int) -> int:
    lower = math.ceil((1 + np.sqrt(1 + 8 * num_edges)) / 2)
    upper = num_edges + 1
    return lower, upper


def critical_K_r(num_edges: int) -> int:
    lower_nodes, upper_nodes = critical_nodes(num_edges)
    return max(
        [
            1 / (1 - 2 / (num_node**2) * num_edges)
            for num_node in range(lower_nodes, upper_nodes)
        ]
    )


def Turan_edges(num_nodes, r):
    return (1 - 1 / r) * num_nodes**2 / 2


if __name__ == "__main__":
    print(Turan_edges(7, 5))
