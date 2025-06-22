from U_stats.statistics.classical import (
    spearman_rho,
    bergsma_dassios_t,
    hoeffding_d,
    set_backend,
)
from scipy.stats import spearmanr
import numpy as np
from time import time
from itertools import permutations

from U_stats import ustat


def main():
    import torch

    n = 1000
    p = 50
    X = np.random.rand(n, p)
    start_time = time()
    result_scipy, _ = spearmanr(X, axis=0)

    end_time = time()
    print(f"Time taken by scipy's spearmanr: {end_time - start_time:.4f} seconds")
    X = torch.tensor(X, dtype=torch.float32)
    start_time = time()
    result_custom = spearman_rho(X)
    end_time = time()
    result_custom = result_custom.numpy()
    print(f"Time taken by custom spearman_rho: {end_time - start_time:.4f} seconds")
    print("Spearman's rho from scipy:\n", result_scipy)
    print("Spearman's rho from custom implementation:\n", result_custom)
    print("Difference:\n", np.max(np.abs(result_scipy - result_custom)))


if __name__ == "__main__":
    set_backend("torch")
    main()
