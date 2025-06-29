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

    def __init__(self, expression: Expression):
        self.expression = UExpression(expression)
        self.order = self.expression.order
        self.shape = self.expression.shape
   
    def calculate(
        self, tensors: List[np.ndarray], average=True, path_method="double-greedy-degree-then-fill", _dediag=True, _einsum=False
    ) -> float:
        """Calculate the U statistics of a list of kernel matrices(tensors)
        with particular expression.

        Args:
            tensors: List[np.ndarray], a list of kernel matrices
            average: bool, whether to average the result
            path_method: str, path optimization method
            _dediag: bool, whether to use non-diagonal calculation
            _einsum: bool, whether to use direct einsum for tensor contraction

        Returns:
            float, the U statistics of the kernel matrices
        """
        tensors = self._initalize_tensor_dict(tensors, self.shape)
        self._validate_inputs(tensors, self.shape)
        n_samples = tensors[0].shape[0]
        
        # Apply dediagonalization if needed
        if _dediag:
            tensors = get_backend().dediag_tensors(tensors, n_samples)
        
        result = 0
        
        # Choose subexpressions based on _dediag flag
        if _dediag:
            subexpressions = self.expression.non_diag_subexpressions()
        else:
            subexpressions = self.expression.subexpressions()
        
        # Calculate result based on _einsum flag
        for weight, subexpression in subexpressions:
            if _einsum:
                # Use einsum method
                result += weight * TensorContractionCalculator._tensor_contract(
                    self, tensors.copy(), expression=subexpression, _einsum=True
                )
            else:
                # Use path-based method
                path, _ = subexpression.path(path_method)
                result += weight * TensorContractionCalculator._tensor_contract(
                    self, tensors.copy(), computing_path=path, _einsum=False
                )
        
        # Apply averaging if needed
        if average:
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
