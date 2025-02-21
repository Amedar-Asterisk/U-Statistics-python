from ..tensor_contraction.state import TensorContractionState
from ..tensor_contraction.calculator import TensorContractionCalculator
from typing import List, Union, Hashable, Generator, Tuple
from statistics.U2V import get_all_partitions, partition_weight
from U_stats.utils import standardize_indexes
import numpy as np
import itertools


class UMode(TensorContractionState):
    def __init__(self, mode: List[Union[List[Hashable], str]]):
        super().__init__(mode)

    def submode(self, partition: List[set]) -> Tuple[float, "UMode"]:
        new_mode = self.copy()
        mapping = {}
        for s in partition:
            representative = s.pop()
            for element in s:
                if element != representative:
                    mapping[element] = representative

        for pos in range(len(new_mode._data)):
            pair = new_mode._data[pos]
            new_pair = "".join(mapping.get(index, index) for index in pair)
            new_mode._data[pos] = new_pair

        for orig, rep in mapping.items():
            new_mode._index_table.merge(orig, rep)

        return partition_weight(partition), new_mode

    def submodes(self) -> Generator["UMode", None, None]:
        partitions = get_all_partitions(self.indexes)
        for partition in partitions:
            yield self.submode(partition)


class UStatsCalculator(TensorContractionCalculator):
    """
    A class for calculating the U statistics of a list of kernel matrices(tensors) with particular mode.
    """

    def __init__(self, mode: List[Union[List[Hashable], str]], summor: str = "numpy"):
        """
        Initialize UStatsCalculator with specified tensor contraction backend.

        Args:
            summor: str, either "numpy" or "torch"
        """
        TensorContractionCalculator.__init__(self, summor)
        self.mode = UMode(mode)
        self.order = self.mode.order
        self.shape = self.mode.shape

    def calculate(
        self, tensors: List[np.ndarray], average=True, _init=True, _validate=True
    ) -> float:
        """
        Calculate the U statistics of a list of kernel matrices(tensors) with particular mode.

        Args:
            tensors: List[np.ndarray], a list of kernel matrices

        Returns:
            float, the U statistics of the kernel matrices
        """
        if _init:
            tensors = self._initalize_tensor_dict(tensors, self.shape)
        if _validate:
            self._validate_inputs(tensors, self.shape)
        result = 0
        for weight, submode in self.mode.submodes():
            result += weight * TensorContractionCalculator._tensor_contract(
                self, tensors, submode, _init=False, _validate=False
            )
        if average:
            n_samples = tensors[0].shape[0]
            return result / np.prod(range(n_samples, n_samples - self.order, -1))
        return result


def U_stats(
    tensors: List[np.ndarray], mode: List[Union[List[Hashable], str]], average=True
) -> float:
    """
    Calculate the U statistics of a list of kernel matrices(tensors) with particular mode.

    Args:
        tensors: List[np.ndarray], a list of kernel matrices

    Returns:
        float, the U statistics of the kernel matrices
    """
    return UStatsCalculator(mode).calculate(tensors, average)


def U_stats_loop(tensors: List[np.ndarray], mode: List[List[int]]) -> float:
    nt = len(tensors)
    ns = tensors[0].shape[0]
    mode = standardize_indexes(mode)
    order = len(set(itertools.chain(*mode)))

    # Calculate the number of permutations beforehand
    num_perms = np.prod(np.arange(ns, ns - order, -1))
    total_sum = 0.0

    for indices in itertools.permutations(range(ns), order):
        product = 1.0
        for i in range(nt):
            # Correctly mapping indices to tensor dimensions
            current_indices = tuple(indices[j] for j in mode[i])
            product *= tensors[i][current_indices]
        total_sum += product

    return total_sum / num_perms
