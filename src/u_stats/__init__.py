"""
U-Statistics Python Package
===========================

A Python package for efficient computation of U-statistics and V-statistics
via tensor contraction.

This package provides:
- Efficient computation of U-statistics and V-statistics
- Support for multiple tensor backends (NumPy, etc.)
- Optimized tensor contraction algorithms
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
__version__ = "0.6.0"
__description__ = "A Python package for efficient computation of U-statistics and V-statistics via tensor contraction."  # noqa: E501
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

    Args:
        tensors: List of input tensors (numpy arrays)
        expression: Tensor contraction expression. Can be:
            - String: Einstein summation notation
            - Tuple: (Inputs, Outputs) specification
            - Inputs: Input specification only
        average: Whether to compute average (True) or sum (False)
        optimize: Optimization strategy for tensor contraction. Accepts the same
            values as opt_einsum.contract() including 'greedy', 'optimal', 'dp',
            'branch-2', 'branch-all', or a custom path specification.

        **kwargs: Additional keyword arguments passed to the computation

    Returns:
        Computed V-statistic value

    Example:
        >>> import numpy as np
        >>> from u_stats import vstat
        >>> x = np.random.randn(100, 5)
        >>> y = np.random.randn(100, 5)
        >>> result = vstat([x, y], "ij,ij->")
    """
    return VStats(expression=expression).calculate(
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

    Args:
        tensors: List of input tensors (numpy arrays)
        expression: Tensor contraction expression. Can be:
            - String: Einstein summation notation
            - Tuple: (Inputs, Outputs) specification
            - Inputs: Input specification only
        average: Whether to compute average (True) or sum (False)
        optimize: Optimization strategy for tensor contraction. Accepts the same
            values as opt_einsum.contract() including 'greedy', 'optimal', 'dp',
            'branch-2', 'branch-all', or a custom path specification.
        _dediag: Whether to remove diagonal terms (True by default for U-statistics)
        **kwargs: Additional keyword arguments passed to the computation

    Returns:
        Computed U-statistic value

    Example:
        >>> import numpy as np
        >>> from u_stats import ustat
        >>> x = np.random.randn(100, 5)
        >>> y = np.random.randn(100, 5)
        >>> result = ustat([x, y], "ij,ij->")
    """
    return UStats(expression=expression).calculate(
        tensors=tensors, average=average, optimize=optimize, _dediag=_dediag, **kwargs
    )
