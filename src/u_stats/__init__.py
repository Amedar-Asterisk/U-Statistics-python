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
__version__ = "0.7.0"
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
from .statistics import UStats, VStats, U_stats_loop
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
    Compute V-statistics from input tensors.

    V-statistics are generalizations of sample moments that involve averaging
    over all possible combinations of observations.

    Parameters:
        tensors (List[np.ndarray]): List of input tensors (numpy arrays).
        expression (str | Tuple[Inputs, Outputs] | Inputs): Tensor contraction
            expression. Can be:
            - String: Einstein summation notation
            - Tuple: (Inputs, Outputs) specification
            - Inputs: Input specification only
        average (bool, optional): Whether to compute average (True) or sum (False).
            Defaults to True.
        optimize (str, optional): Optimization strategy for tensor contraction.
            Accepts the same values as opt_einsum.contract() including 'greedy',
            'optimal', 'dp', 'branch-2', 'branch-all', or a custom path
            specification. Defaults to "greedy".
        **kwargs: Additional keyword arguments passed to the computation.

    Returns:
        float: Computed V-statistic value.

    Example:
        >>> import numpy as np
        >>> from u_stats import vstat
        >>> x = np.random.randn(100, 5)
        >>> y = np.random.randn(100, 5)
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
    Compute U-statistics from input tensors.

    U-statistics are unbiased estimators based on averaging over all possible
    combinations of distinct observations (no replacement).

    Parameters:
        tensors (List[np.ndarray]): List of input tensor with dtype np.ndarray
            or torch.Tensor. Each tensor is the tensorization of the decomposition
            factors of the U-statistic's kernel, as an example, if the kernel
            h = h_1 h_2 ... h_K and all h_k is defined on \bbX^2, X is a
            list of samples from \bbX, then
                    T^{(k)}_ij = h_k(X_i, X_j),
            where X_i, X_j is i-th and j-th sample in X.
        expression (str | Tuple[Inputs, Outputs] | Inputs): The Einstein summation
            expression defining the U-statistic structure, which define the
            decomposition form of the U-statistic's kernel. Can be:
            - String: Einstein summation notation
            - Tuple: (Inputs, Outputs) specification
            - Inputs: Input specification only
        average (bool, optional): Whether to compute average (True) or sum (False).
            Defaults to True.
        optimize (str, optional): Optimization strategy for tensor contraction.
            Accepts the same values as opt_einsum.contract() including 'greedy',
            'optimal', 'dp', 'branch-2', 'branch-all', or a custom path
            specification. Defaults to "greedy".
        _dediag (bool, optional): Whether to remove diagonal terms (True by
            default for U-statistics). Defaults to True.
        **kwargs: Additional keyword arguments passed to the computation.

    Returns:
        float: Computed U-statistic value.

    Example:
        >>> import numpy as np
        >>> from u_stats import ustat
        >>> x = np.random.randn(100, 5)
        >>> y = np.random.randn(100, 5)
        >>> result = ustat([x, y], "ij,ij->")
    """
    return UStats(expression=expression).compute(
        tensors=tensors, average=average, optimize=optimize, _dediag=_dediag, **kwargs
    )
