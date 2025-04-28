from ..tensor_contraction import TensorContractionCalculator
from ..tensor_contraction import TensorContractionState
from typing import List, Union, Hashable
import numpy as np

__all__ = ["VStatsCalculator", "V_stats"]


class VStatsCalculator(TensorContractionCalculator):
    """
    A class for calculating the statistics of a list of kernel matrices(tensors) with particular mode.
    """

    def __init__(self, mode: List[Union[List[Hashable], str]], summor: str = "numpy"):
        """
        Initialize VStatsCalculator with specified tensor contraction backend.

        Args:
            summor: str, either "numpy" or "torch"
        """
        super().__init__(summor)
        self.mode = TensorContractionState(mode)
        self.order = self.mode.order
        self.shape = self.mode.shape

    def calculate(
        self, tensors: List[np.ndarray], average=True, _init=True, _validate_=True
    ) -> float:
        """
        Calculate the statistics of a list of kernel matrices(tensors) with particular mode.

        Args:
            tensors: List[np.ndarray], a list of kernel matrices

        Returns:
            float, the V statistics of the kernel matrices
        """
        mode = self.mode.copy()
        tensors = tensors.copy()
        n_samples = tensors[0].shape[0]
        if _init:
            tensors = TensorContractionCalculator._initalize_tensor_dict(
                self, tensors, self.shape
            )
        if _validate_:
            TensorContractionCalculator._validate_inputs(self, tensors, self.shape)

        result = TensorContractionCalculator._tensor_contract(self, tensors, mode)

        if average:
            return result / (n_samples**self.order)
        return result


def V_stats(
    tensors: List[np.ndarray], mode: List[Union[List[Hashable], str]], average=True
) -> float:
    """
    Calculate the V statistics of a list of kernel matrices(tensors) with particular mode.

    Args:
        tensors: List[np.ndarray], a list of kernel matrices

    Returns:
        float, the V statistics of the kernel matrices
    """
    return VStatsCalculator(mode).calculate(tensors, average)
