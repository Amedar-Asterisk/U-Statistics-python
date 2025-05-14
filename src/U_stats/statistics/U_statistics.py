from ..tensor_contraction.path import TensorExpression, NestedHashableList
from ..tensor_contraction.calculator import TensorContractionCalculator
from typing import List, Union, Hashable, Generator, Tuple
from .U2V import get_all_partitions, partition_weight
from ..utils import standardize_indexes
import numpy as np
import itertools

__all__ = ["UStatsCalculator", "U_stats"]


class UExpression(TensorExpression):

    def __init__(self, mode: NestedHashableList):
        super().__init__(mode)

    @property
    def order(self) -> int:
        return len(self.indices)

    def submode(self, partition: List[set]) -> Tuple[float, "UExpression"]:
        new_mode = self.copy()
        mapping = {}
        weight = partition_weight(partition)

        for s in partition:
            representative = min(s)
            for element in s:
                if element != representative:
                    mapping[element] = representative
        for pos in new_mode._pair_dict:
            pair = new_mode[pos]
            new_pair = [mapping.get(index, index) for index in pair]
            new_mode._pair_dict[pos] = new_pair

        for orig, rep in mapping.items():
            new_mode._index_table.merge(orig, rep)
        return weight, new_mode

    def submodes(self) -> Generator[Tuple[float, "UExpression"], None, None]:
        partitions = get_all_partitions(self._index_table.indices)
        for partition in partitions:
            yield self.submode(partition)


class UStatsCalculator(TensorContractionCalculator):
    """
    A class for calculating the U statistics of a list of kernel matrices(tensors) with particular mode.
    """

    def __init__(self, mode: NestedHashableList, summor: str = "numpy"):
        """
        Initialize UStatsCalculator with specified tensor contraction backend.

        Args:
            summor: str, either "numpy" or "torch"
        """
        TensorContractionCalculator.__init__(self, summor)
        self.mode = UExpression(mode)
        self.order = self.mode.order
        self.shape = self.mode.shape

    def calculate(
        self, tensors: List[np.ndarray], average=True, path_method="greedy"
    ) -> float:
        """
        Calculate the U statistics of a list of kernel matrices(tensors) with particular mode.

        Args:
            tensors: List[np.ndarray], a list of kernel matrices

        Returns:
            float, the U statistics of the kernel matrices
        """
        tensors = self._initalize_tensor_dict(tensors, self.shape)
        self._validate_inputs(tensors, self.shape)
        result = 0
        submodes = self.mode.submodes()
        for weight, submode in submodes:
            path, cost = submode.path("greedy")
            result += weight * TensorContractionCalculator._tensor_contract(
                self, tensors.copy(), path
            )
        if average:
            n_samples = tensors[0].shape[0]
            return result / np.prod(range(n_samples, n_samples - self.order, -1))
        return result


def U_stats(
    tensors: List[np.ndarray],
    mode: NestedHashableList,
    average=True,
    summor: str = "numpy",
) -> float:
    """
    Calculate the U statistics of a list of kernel matrices(tensors) with particular mode.

    Args:
        tensors: List[np.ndarray], a list of kernel matrices
        mode: List[Union[List[Hashable], str]], the mode of the U statistics
        average: bool, whether to average the U statistics
        summor: str, either "numpy" or "torch"

    Returns:
        float, the U statistics of the kernel matrices
    """
    return UStatsCalculator(mode, summor=summor).calculate(tensors, average)


def U_stats_loop(tensors: List[np.ndarray], mode: List[List[int]]) -> float:
    nt = len(tensors)
    ns = tensors[0].shape[0]
    mode = standardize_indexes(mode)
    order = len(set(itertools.chain(*mode)))

    num_perms = np.prod(np.arange(ns, ns - order, -1))
    total_sum = 0.0

    for indices in itertools.permutations(range(ns), order):
        product = 1.0
        for i in range(nt):
            current_indices = tuple(indices[j] for j in mode[i])
            product *= tensors[i][current_indices]
        total_sum += product

    return total_sum / num_perms
