from ..tensor_contraction.calculator import TensorContractionCalculator, BACKEND
from ..tensor_contraction.path import TensorExpression, NestedHashableList
from typing import List, Union, Hashable
import numpy as np


class VExpression(TensorExpression):

    def __init__(self, mode: NestedHashableList):
        super().__init__(mode)
        self._path = {}
        self._cost = {}

    @property
    def order(self) -> int:
        return len(self.indices)

    def path(self, method: str = "greedy"):
        if method in self._path:
            return self._path[method]
        else:
            self._path[method], self._cost[method] = TensorExpression.path(self, method)
            return self._path[method]


class VStatsCalculator(TensorContractionCalculator):
    """
    A class for calculating the statistics of a list of kernel matrices(tensors) with particular mode.
    """

    def __init__(self, mode: NestedHashableList, summor: str = "numpy"):
        """
        Initialize VStatsCalculator with specified tensor contraction backend.

        Args:
            summor: str, either "numpy" or "torch"
        """
        super().__init__(summor)
        self.mode = VExpression(mode)
        self.shape = self.mode.shape
        self.order = len(self.mode.order)

    def calculate(
        self,
        tensors: List[np.ndarray],
        average=True,
        path_method: str = "greedy",
    ) -> float:
        n_samples = tensors[0].shape[0]
        tensors = TensorContractionCalculator._initalize_tensor_dict(
            self, tensors, self.shape
        )
        TensorContractionCalculator._validate_inputs(self, tensors, self.shape)
        path, _ = self.mode.path(path_method)
        result = TensorContractionCalculator._tensor_contract(self, tensors, path)
        if average:
            return result / (n_samples**self.order)
        return result


def V_stats(
    tensors: List[np.ndarray], mode: NestedHashableList, average=True, summor="numpy"
) -> float:
    """
    Calculate the V statistics of a list of kernel matrices(tensors) with particular mode.

    Args:
        tensors: List[np.ndarray], a list of kernel matrices

    Returns:
        float, the V statistics of the kernel matrices
    """
    return VStatsCalculator(mode, summor=summor).calculate(tensors, average)
