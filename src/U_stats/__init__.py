__title__ = "u_stats"
__version__ = "0.4.1"
__description__ = "A Python package for efficient computation of U-statistics and V-statistics via tensor contraction."  # noqa: E501
__author__ = "Zhang Ruiqi"
__author_email__ = "zrq1706@outlook.com"
__license__ = "MIT"

__all__ = [
    "vstat",
    "ustat",
    "analyze_expression",
    "complexity",
    "TensorExpression",
    "UStatsCalculator",
    "VStatsCalculator",
    "U_stats_loop",
    "set_backend",
    "Backend",
    "available_methods",
]
from .tensor_contraction.path import TensorExpression
from .statistics import UStatsCalculator, VStatsCalculator, U_stats_loop
from ._utils import Expression, PathInfo
from ._utils import set_backend, Backend
from typing import List, Tuple
import numpy as np


def vstat(
    tensors: List[np.ndarray],
    expression: Expression,
    average: bool = True,
    path_method: str = "double-greedy-degree-then-fill",
):
    return VStatsCalculator(expression=expression).calculate(
        tensors=tensors, average=average, path_method=path_method
    )


def ustat(
    tensors: List[np.ndarray],
    expression: Expression,
    average: bool = True,
    path_method: str = "double-greedy-degree-then-fill",
    _dediag: bool = True,
) -> float:
    return (
        UStatsCalculator(expression=expression).caculate_non_diag(
            tensors=tensors, average=average, path_method=path_method
        )
        if _dediag
        else UStatsCalculator(expression=expression).calculate(
            tensors=tensors, average=average, path_method=path_method
        )
    )


def analyze_expression(
    expression: Expression, path_method: str, size: int = 10**4
) -> PathInfo:
    """Analyze the expression and return the path and cost."""
    exp = TensorExpression(expression=expression)
    path, _ = exp.path(path_method)
    info = exp.analyze_path(path, size=size, optimize="auto")
    return info


def complexity(
    expression: Expression, path_method: str, size: int = 10**4
) -> Tuple[int, int, int]:
    """Analyze the expression and return the path and cost."""
    info = analyze_expression(expression, path_method=path_method, size=size)
    rank, flops, memory = max(info.scale_list), info.opt_cost, info.largest_intermediate
    return rank, flops, memory


available_methods = list(TensorExpression._METHOD_.keys())
