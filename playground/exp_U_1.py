from typing import Tuple, List, Union, Any
import numpy as np
import time
from init import *
from U_stats.statistics.U_statistics import U_stats, UStatsCalculator
from U_stats.statistics.U_statistics import U_stats_loop


def produce_data(
    n: int, p: int, loc: float = 0, scale: float = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for U-statistics experiment.

    Args:
        n: Number of samples.
        p: Number of features.
        loc: Mean of the normal distribution. Defaults to 0.
        scale: Standard deviation of the normal distribution. Defaults to 1.

    Returns:
        Tuple containing:
            - X: Generated feature matrix of shape (n, p)
            - A: Binary outcome vector of shape (n,)
    """
    X = np.random.normal(loc=loc, scale=scale, size=(n, p))
    s = round(np.sqrt(p))
    bound = np.sqrt(3 / p)
    alpha = np.random.uniform(-bound, bound, size=p)

    z = X @ alpha
    prob = 1 / (1 + np.exp(-z))
    A = np.random.binomial(1, prob, size=n)
    return X, A


@timer
def get_input_tensor(n: int, kappa: float = 0.5):
    """Create kernel matrix and treatment vector for U-statistics computation.

    Args:
        n: Number of samples.
        kappa: Scale factor for number of features (p = kappa * n).

    Returns:
        Tuple containing:
            - Ker: Kernel matrix of shape (n, n)
            - A: Treatment vector of shape (n,)
    """
    p = int(kappa * n)
    X, A = produce_data(n, p)
    Ker = X @ X.T
    return Ker, A


def get_tensors(m: int, Ker: np.ndarray, A: np.ndarray) -> List[np.ndarray]:
    """Assemble input tensors for U-statistics computation.

    Args:
        m: Number of tensors to generate.
        Ker: Kernel matrix.
        A: Treatment vector.

    Returns:
        List of m tensors where first and last elements are treatment vectors,
        and middle elements are kernel matrices.
    """
    outputs = []
    outputs.append(A)
    for _ in range(m - 1):
        outputs.append(Ker)
    outputs.append(A)
    return outputs


def test_mode(m: int) -> List[Union[int, Tuple[int, int]]]:
    """Generate mode configuration for U-statistics testing.

    Args:
        m: Number of tensors.

    Returns:
        List of modes where first and last elements are single integers,
        and middle elements are pairs of consecutive integers.
    """
    outputs = []
    for i in range(m + 1):
        if i == 0:
            outputs.append([i])
        elif i == m:
            outputs.append([i - 1])
        else:
            outputs.append([i - 1, i])
    return outputs


def prepare_tensors_mode(m: int, Ker: np.ndarray, A: np.ndarray):
    """Prepare input tensors and mode for U-statistics computation.

    args:
        m: Number of tensors.
        Ker: Kernel matrix.
        A: Treatment vector.

    Returns:
        Tuple containing:
            - inputs: List of input tensors for U-statistics computation.
            - mode: Mode configuration for U-statistics testing.
    """
    inputs = get_tensors(m, Ker, A)
    mode = test_mode(m)
    return inputs, mode


@timer
def test_our(tensors, mode, summor="numpy"):
    return U_stats(tensors, mode, summor=summor)


@timer
def test_loop(tensors, mode):
    return U_stats_loop(tensors, mode)


@timer
def test_nodiag(tensors, mode, summor="numpy"):
    tensors = tensors.copy()
    caculator = UStatsCalculator(mode, summor=summor)
    return caculator.caculate_non_diag(tensors)


def test1(n, m, summor="numpy"):
    tensors, assemble_time = get_input_tensor(n)
    inputs, mode = prepare_tensors_mode(m, *tensors)
    result_our, time_our = test_our(inputs, mode, summor=summor)
    print("result_our:", result_our)
    print("assemble_time:", assemble_time)
    print("time_our:", time_our)


def test2(n, m, summor="numpy"):
    tensors, assemble_time = get_input_tensor(n)
    inputs, mode = prepare_tensors_mode(m, *tensors)
    result_loop, time_loop = test_loop(inputs, mode)
    result_our, time_our = test_our(inputs, mode, summor=summor)
    result_nodiag, time_nodiag = test_nodiag(inputs, mode)
    print("assemble_time:", assemble_time)
    print("result_our:", result_our)
    print("result_loop:", result_loop)
    print("result_nodiag:", result_nodiag)
    print("time_our:", time_our)
    print("time_nodiag:", time_nodiag)
    print("time_loop:", time_loop)
    print(
        "relative error(ours):",
        np.abs(result_our - result_loop) / np.abs(result_loop),
    )
    print(
        "relative error(nodiag):",
        np.abs(result_nodiag - result_loop) / np.abs(result_loop),
    )


def test3(n, m, summor="numpy"):
    tensors, assemble_time = get_input_tensor(n)
    inputs, mode = prepare_tensors_mode(m, *tensors)
    result_our, time_our = test_our(inputs, mode, summor=summor)
    result_nodiag, time_nodiag = test_nodiag(inputs, mode, summor=summor)
    print("assemble_time:", assemble_time)
    print("result_our:", result_our)
    print("result_nodiag:", result_nodiag)
    print(
        "relative error:",
        np.abs(result_our - result_nodiag) / np.abs(result_our),
    )
    print("time_our:", time_our)
    print("time_nodiag:", time_nodiag)


if __name__ == "__main__":
    n = 1000
    m = 8
    print("torch")
    test3(n, m, summor="torch")
    # print("numpy")
    # test3(n, m, summor="numpy")
