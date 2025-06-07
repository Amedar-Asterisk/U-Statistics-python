from ..tensor_contraction.path import TensorExpression
from ..tensor_contraction.calculator import TensorContractionCalculator
from typing import List, Hashable, Generator, Tuple, Dict, Set
from functools import cached_property
from .U2V import get_all_partitions, partition_weight, get_all_partitions_nonconnected
from .._utils import Expression, standardize_indices
import numpy as np
import itertools

__all__ = [
    "UStatsCalculator",
    "U_stats_loop",
]


class UExpression(TensorExpression):

    def __init__(self, expression: Expression):
        super().__init__(expression)

    @property
    def order(self) -> int:
        return len(self.indices)

    def subexpression(self, partition: List[set]) -> Tuple[float, "UExpression"]:
        new_expression = self.copy()
        mapping = {}
        weight = partition_weight(partition)

        for s in partition:
            representative = min(s)
            for element in s:
                if element != representative:
                    mapping[element] = representative
        for pos in new_expression._pair_dict:
            pair = new_expression[pos]
            new_pair = [mapping.get(index, index) for index in pair]
            new_expression._pair_dict[pos] = new_pair

        for orig, rep in mapping.items():
            new_expression._index_table.merge(orig, rep)
        return weight, new_expression

    @cached_property
    def _adj_list(self) -> Dict[Hashable, Set[Hashable]]:
        adj_list = {}
        for index in self.indices:
            adj_list[index] = set()
            for pair in self._pair_dict.values():
                if index in pair:
                    adj_list[index].update(pair)
            adj_list[index].discard(index)
        return adj_list

    def subexpressions(self) -> Generator[Tuple[float, "UExpression"], None, None]:
        partitions = get_all_partitions(self._index_table.indices)
        for partition in partitions:
            yield self.subexpression(partition)

    def non_diag_subexpressions(
        self,
    ) -> Generator[Tuple[float, "UExpression"], None, None]:
        partitions = get_all_partitions_nonconnected(self._adj_list)
        for partition in partitions:
            weight, subexpression = self.subexpression(partition)
            yield weight, subexpression


class UStatsCalculator(TensorContractionCalculator):
    """A class for calculating the U statistics of a list of kernel
    matrices(tensors) with particular expression."""

    def __init__(self, expression: Expression, summor: str = "numpy"):
        """Initialize UStatsCalculator with specified tensor contraction
        backend.

        Args:
            summor: str, either "numpy" or "torch"
        """
        TensorContractionCalculator.__init__(self, summor)
        self.expression = UExpression(expression)
        self.order = self.expression.order
        self.shape = self.expression.shape

    def calculate(
        self, tensors: List[np.ndarray], average=True, path_method="greedy"
    ) -> float:
        """Calculate the U statistics of a list of kernel matrices(tensors)
        with particular expression.

        Args:
            tensors: List[np.ndarray], a list of kernel matrices

        Returns:
            float, the U statistics of the kernel matrices
        """
        tensors = self._initalize_tensor_dict(tensors, self.shape)
        self._validate_inputs(tensors, self.shape)
        result = 0
        subexpressions = self.expression.subexpressions()
        for weight, subexpression in subexpressions:
            path, cost = subexpression.path(path_method)
            result += weight * TensorContractionCalculator._tensor_contract(
                self, tensors.copy(), path
            )
        if average:
            n_samples = tensors[0].shape[0]
            return result / np.prod(range(n_samples, n_samples - self.order, -1))
        return result

    def caculate_non_diag(
        self, tensors: List[np.ndarray], average=True, path_method="greedy"
    ) -> float:
        """Calculate the U statistics of a list of kernel matrices(tensors)
        with particular expression.

        Args:
            tensors: List[np.ndarray], a list of kernel matrices

        Returns:
            float, the U statistics of the kernel matrices
        """
        tensors = self._initalize_tensor_dict(tensors, self.shape)
        self._validate_inputs(tensors, self.shape)
        n_samples = tensors[0].shape[0]
        if self.summor_name == "numpy":
            tensors = self._mask_tensors(tensors, n_samples)
        elif self.summor_name == "torch":
            tensors = self._torch_mask_tensors(tensors, n_samples, self.device)

        result = 0
        subexpressions = self.expression.non_diag_subexpressions()
        for weight, subexpression in subexpressions:
            path, cost = subexpression.path(path_method)
            result += weight * TensorContractionCalculator._tensor_contract(
                self, tensors.copy(), computing_path=path
            )
        if average:
            return result / np.prod(range(n_samples, n_samples - self.order, -1))
        return result

    @staticmethod
    def _mask_tensors(
        tensors: Dict[int, np.ndarray], sample_size: int
    ) -> Dict[int, np.ndarray]:
        shapes = {index: tensor.ndim for index, tensor in tensors.items()}
        for index, ndim in shapes.items():
            if ndim > 1:
                mask_total = np.zeros((sample_size,) * ndim, dtype=bool)
                for i, j in itertools.combinations(range(ndim), 2):
                    mask = UStatsCalculator._mask_tensor(ndim, sample_size, i, j)
                    mask_total |= mask
                mask_total = np.logical_not(mask_total)
                tensors[index] = tensors[index] * mask_total
        return tensors

    @staticmethod
    def _mask_tensor(ndim: int, dim: int, index1: int, index2: int) -> np.ndarray:
        shape1 = [1] * ndim
        shape1[index1] = dim
        shape2 = [1] * ndim
        shape2[index2] = dim

        idx1 = np.arange(dim).reshape(shape1)
        idx2 = np.arange(dim).reshape(shape2)
        mask = idx1 == idx2
        mask = np.broadcast_to(mask, (dim,) * ndim)
        return mask


def U_stats_loop(tensors: List[np.ndarray], expression: List[List[int]]) -> float:
    nt = len(tensors)
    ns = tensors[0].shape[0]
    expression = standardize_indices(expression)
    order = len(set(itertools.chain(*expression)))

    num_perms = np.prod(np.arange(ns, ns - order, -1))
    total_sum = 0.0

    for indices in itertools.permutations(range(ns), order):
        product = 1.0
        for i in range(nt):
            current_indices = tuple(indices[j] for j in expression[i])
            product *= tensors[i][current_indices]
        total_sum += product

    return total_sum / num_perms
