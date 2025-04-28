from typing import Tuple, List, Union, Any
import numpy as np
import time
from init import *
from U_stats.statistics.U_statistics import U_stats


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


def get_input_tensor(n: int, kappa: float) -> Tuple[np.ndarray, np.ndarray]:
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


def U_stats_single(m: int, Ker: np.ndarray, A: np.ndarray) -> Tuple[Any, float]:
    """Compute U-statistics for a single tensor.

    Args:
        m: Number of tensors.
        Ker: Kernel matrix.
        A: Treatment vector.

    Returns:
        Computed U-statistics result.
        Computing time.
    """
    inputs = get_tensors(m, Ker, A)
    mode = test_mode(m)
    time1 = time.time()
    result = U_stats(inputs, mode)
    time2 = time.time()
    compute_time = time2 - time1
    return result, compute_time


def test(n: int, m: int | List[int], kappa: float = 0.5) -> Tuple[Any, float, float]:
    """Run U-statistics test and measure execution time.

    Args:
        n: Number of samples.
        m: Number of tensors.
        kappa: Scale factor for number of features.

    Returns:
        Tuple containing:
            - result: Output of U-statistics computation
            - assemble_time: Time taken to assemble input tensors
            - compute_time: Time taken to compute U-statistics
    """
    try:
        time1 = time.time()
        Ker, A = get_input_tensor(n, kappa)
        time2 = time.time()
        assemble_time = time2 - time1

        if isinstance(m, int):
            result, compute_time = U_stats_single(m, Ker, A)
        elif isinstance(m, list):
            result = []
            compute_time = []
            for i in m:
                res, time3 = U_stats_single(i, Ker, A)
                result.append(res)
                compute_time.append(time3)
        return result, assemble_time, compute_time
    except Exception as e:
        print(f"Testing Error: {e}")
        return None, None, None
