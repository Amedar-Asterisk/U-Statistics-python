__title__ = "u_stats"
__version__ = "0.5.0"
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
):
    return VStats(expression=expression).calculate(
        tensors=tensors, average=average, optimize=optimize
    )


def ustat(
    tensors: List[np.ndarray],
    expression: str | Tuple[Inputs, Outputs] | Inputs,
    average: bool = True,
    optimize: str = "greedy",
    _dediag: bool = True,
) -> float:
    return UStats(expression=expression).calculate(
        tensors=tensors, average=average, optimize=optimize, _dediag=_dediag
    )
