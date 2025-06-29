from ..tensor_contraction.path import TensorExpression
from ..tensor_contraction.calculator import TensorContractionCalculator
from typing import List, Hashable, Generator, Tuple, Dict, Set
from functools import cached_property
from .U2V import get_all_partitions, partition_weight, get_all_partitions_nonconnected
from .._utils import Expression, standardize_indices, get_backend
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

    def subexpression(self, partition: List[set]) -> Tuple[float, TensorExpression]:
        mapping = {}
        weight = partition_weight(partition)
        expression = self.expression
        for s in partition:
            representative = min(s)
            for element in s:
                if element != representative:
                    mapping[element] = representative
        new_expression = []
        for pair in expression:
            new_pair = [mapping.get(index, index) for index in pair]
            new_expression.append(new_pair)
        return (
            weight,
            TensorExpression(new_expression),
        )

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

    def __init__(self, expression: Expression):
        self.expression = UExpression(expression)
        self.order = self.expression.order
        self.shape = self.expression.shape

    def calculate(
        self,
        tensors: List[np.ndarray],
        average=True,
        path_method="greedy-fill-in",
        dediag: bool = True,
        use_einsum: bool = False,
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
        if dediag:
            n_samples = tensors[0].shape[0]
            tensors = get_backend().dediag_tensors(tensors, n_samples)
            subexpressions = self.expression.non_diag_subexpressions()
        else:
            subexpressions = self.expression.subexpressions()

        if use_einsum:
            tensors = [tensors[i] for i in range(len(tensors))]

        for weight, subexpression in subexpressions:
            if use_einsum:
                result += weight * get_backend().einsum(str(subexpression), *tensors)
            else:
                path, _ = subexpression.path(path_method)
                result += weight * TensorContractionCalculator._tensor_contract(
                    self, tensors.copy(), path
                )
        if average:
            n_samples = tensors[0].shape[0]
            return result / get_backend().prod(
                range(n_samples, n_samples - self.order, -1)
            )
        return result


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
