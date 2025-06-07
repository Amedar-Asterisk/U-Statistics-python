from ..tensor_contraction.calculator import TensorContractionCalculator
from ..tensor_contraction.path import TensorExpression
from ..utils._typing import Expression
from typing import List
import numpy as np


class VExpression(TensorExpression):

    def __init__(self, expression: Expression):
        super().__init__(expression)
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
    """A class for calculating the statistics of a list of kernel
    matrices(tensors) with particular expression."""

    def __init__(self, expression: Expression, summor: str = "numpy"):
        """Initialize VStatsCalculator with specified tensor contraction
        backend.

        Args:
            summor: str, either "numpy" or "torch"
        """
        super().__init__(summor)
        self.expression = VExpression(expression)
        self.shape = self.expression.shape
        self.order = self.expression.order

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
        path, _ = self.expression.path(path_method)
        result = TensorContractionCalculator._tensor_contract(self, tensors, path)
        if average:
            return result / (n_samples**self.order)
        return result


def V_stats(
    tensors: List[np.ndarray], expression: Expression, average=True, summor="numpy"
) -> float:
    """Calculate the V statistics of a list of kernel matrices(tensors) with
    particular expression.

    Args:
        tensors: List[np.ndarray], a list of kernel matrices

    Returns:
        float, the V statistics of the kernel matrices
    """
    return VStatsCalculator(expression, summor=summor).calculate(tensors, average)
