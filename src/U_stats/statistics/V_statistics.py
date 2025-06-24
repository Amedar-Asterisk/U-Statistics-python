from ..tensor_contraction.calculator import TensorContractionCalculator
from ..tensor_contraction.path import TensorExpression
from .._utils import Expression
from typing import List
import numpy as np

__all__ = ["VStatsCalculator"]


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

    def __init__(self, expression: Expression):
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
        path = self.expression.path(path_method)
        result = TensorContractionCalculator._tensor_contract(self, tensors, path)
        if average:
            return result / (n_samples**self.order)
        return result
