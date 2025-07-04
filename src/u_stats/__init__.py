"""
U-Statistics Python Package
===========================

A Python package for efficient computation of U-statistics
via tensor contraction.

This package provides:
- Efficient computation of U-statistics
- Support for multiple tensor backends (NumPy and Torch)
- Both high-level convenience functions and low-level class interfaces

Main Functions:
    vstat: Compute V-statistics from tensors
    ustat: Compute U-statistics from tensors

Main Classes:
    UStats: Class for U-statistics computation
    VStats: Class for V-statistics computation
    U_stats_loop: Loop-based U-statistics computation

Utilities:
    set_backend: Set the tensor computation backend
    get_backend: Get the current tensor computation backend
"""

__title__ = "u_stats"
__version__ = "0.7.4"
__description__ = "A Python package for efficient computation of U-statistics via tensor contraction."  # noqa: E501
__author__ = "Zhang Ruiqi"
__author_email__ = "zrq1706@outlook.com"
__license__ = "MIT"

__all__ = [
    "vstat",
    "ustat",
    "UStats",
    "VStats",
    "U_stats_loop",
    "set_backend",
    "get_backend",
]
from .statistics import UStats, VStats, u_stats_loop
from ._utils import set_backend, Backend, get_backend
from typing import List, Tuple
from ._utils import Inputs, Outputs
import numpy as np


def vstat(
    tensors: List[np.ndarray],
    expression: str | Tuple[Inputs, Outputs] | Inputs,
    average: bool = True,
    optimize: str = "greedy",
    **kwargs,
) -> float:
    """
    Compute the V-statistic on input tensors and corresponding expression.

    This function performs direct Einstein summation to compute the V-statistic.

    Parameters
    ----------
    tensors : list of np.ndarray
        Input tensors for the V-statistic computation. Each tensor represents
        the tensorization of factors in the V-statistic's kernel. For example,
        if the kernel h = h_1 * h_2 * ... * h_K and each h_k is defined on
        X^d, then T^{(k)}_{i1,i2,...,id} = h_k(X_{i1}, X_{i2}, ..., X_{id}).
    expression : str or tuple or list
        The Einstein summation expression defining the U-statistic structure,
        which define the decomposition form of the U-statistic's kernel

        Supported formats:
        - **String**: Einstein notation (e.g., 'ij,jk->', 'ab,bc,ca->')
        - **Tuple**: (inputs, outputs) where:
            - inputs: sequence of sequences of hashable indices
            - outputs: sequence of hashable output indices
        - **List**: input indices only (output defaults to scalar)
    average : bool, default=True
        Whether to return the averaged V-statistic. If False, returns
        the unscaled sum over all valid index combinations.
    optimize : str, default='greedy'
        Optimization strategy for the einsum contraction path.
        Options include 'greedy', 'optimal', 'dp', 'auto', etc,
        with is the same as the `optimize` parameter in `opt_einsum.contract`.
    **kwargs
        Additional keyword arguments passed to the backend's einsum function.
        Common options include 'optimize' for path optimization strategy.

    Returns
    -------
    float or np.ndarray
        The computed V-statistic value:
        - **float**: For scalar V-statistics (no output indices)
        - **np.ndarray**: For tensor-valued V-statistics (with output indices)

    Example:
        >>> import numpy as np
        >>> from u_stats import vstat
        >>> x = np.random.randn(100, 100)
        >>> y = np.random.randn(100, 100)
        >>> result = vstat([x, y], "ij,ij->")
    """
    return VStats(expression=expression).compute(
        tensors=tensors, average=average, optimize=optimize, **kwargs
    )


def ustat(
    tensors: List[np.ndarray],
    expression: str | Tuple[Inputs, Outputs] | Inputs,
    average: bool = True,
    optimize: str = "greedy",
    _dediag: bool = True,
    **kwargs,
) -> float:
    """
    Compute the U-statistic on input tensors and corresponding expression.

    Parameters
    ----------
    tensors : list of np.ndarray or torch.Tensor
        Input tensors for the V-statistic computation. Each tensor represents
        the tensorization of factors in the V-statistic's kernel. For example,
        if the kernel h = h_1 * h_2 * ... * h_K and each h_k is defined on
        X^d, then T^{(k)}_{i1,i2,...,id} = h_k(X_{i1}, X_{i2}, ..., X_{id}).
    expression : str or tuple or list
        The Einstein summation expression defining the U-statistic structure,
        which define the decomposition form of the U-statistic's kernel

        Supported formats:
        - **String**: Einstein notation (e.g., 'ij,jk->', 'ab,bc,ca->')
        - **Tuple**: (inputs, outputs) where:
            - inputs: sequence of sequences of hashable indices
            - outputs: sequence of hashable output indices
        - **List**: input indices only (output defaults to scalar)
    average : bool, default=True
        Whether to return the averaged U-statistic. If False, returns
        the unscaled sum over all valid index combinations.
    optimize : str, default='greedy'
        Optimization strategy for the einsum contraction path.
        Options include 'greedy', 'optimal', 'dp', 'auto', etc,
        with is the same as the `optimize` parameter in `opt_einsum.contract`.
    _dediag : bool, default=True
        Whether to apply dediagonalization (exclude diagonal terms).
        Automatically disabled for tensor-valued U-statistics.
        Setting to False computes V-statistics instead.
    **kwargs
        Additional keyword arguments passed to opt_einsum.contract().
        Common options include 'optimize' for path optimization strategy.
        ""optimize" can be 'greedy', 'optimal', 'dp', "auto", etc.

    Returns
    -------
    float or np.ndarray
        The computed U-statistic value:
        - **float**: For scalar U-statistics (no output indices)
        - **np.ndarray**: For tensor-valued U-statistics (with output indices)
        (under testing)

    Example:
        >>> import numpy as np
        >>> from u_stats import ustat
        >>> x = np.random.randn(100, 100)
        >>> y = np.random.randn(100, 100)
        >>> result = ustat([x, y], "ij,ij->")
    """
    return UStats(expression=expression).compute(
        tensors=tensors, average=average, optimize=optimize, _dediag=_dediag, **kwargs
    )
