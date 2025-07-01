from typing import List, Generator, Tuple, Dict, Set
from functools import cached_property
from .u2v import (
    get_all_partitions,
    partition_weight,
    get_all_partitions_nonconnected,
    get_adj_list,
)
from .._utils import (
    standardize_indices,
    get_backend,
    numbers_to_letters,
    strlist_to_einsum_eq,
    einsum_eq_to_strlist,
    Inputs,
    Outputs,
)
from .._utils._typing import ComplexityInfo
import numpy as np
import itertools
import warnings
import opt_einsum as oe

__all__ = [
    "UStats",
    "U_stats_loop",
]


class UStats:
    def __init__(self, expression: str | Tuple[Inputs, Outputs] | Inputs):
        if isinstance(expression, str):
            self._ep = expression
            inputs, output = einsum_eq_to_strlist(expression)
        else:
            if isinstance(expression, tuple):
                inputs, output = expression
                inputs = numbers_to_letters(standardize_indices(inputs))
                output = numbers_to_letters(standardize_indices(output))
            else:
                inputs = expression
                output = ""
                inputs = numbers_to_letters(standardize_indices(inputs))
            self._ep = strlist_to_einsum_eq(inputs, output)
        self._reserved_indices = output
        self._inputs = inputs
        self._contracted_indices = set(itertools.chain(*inputs))
        for index in self._reserved_indices:
            self._contracted_indices.discard(index)

    @property
    def expression(self) -> str:
        """Get the expression of the U statistics."""
        return self._ep

    @cached_property
    def _adj_list(self) -> Dict[str, Set[str]]:
        return get_adj_list(self._inputs)

    @cached_property
    def order(self) -> int:
        """Get the order of the U statistics."""
        return len(self._contracted_indices)

    @cached_property
    def input_index_position(self) -> Tuple[int, int]:
        for i, inputs in enumerate(self._inputs):
            for index in self._contracted_indices:
                pos = inputs.find(index)
                if pos != -1:
                    return (i, pos)

    def _get_all_subexpressions(
        self, dediag: bool = True
    ) -> Generator[Tuple[float, str], None, None]:
        if dediag:
            partitions = get_all_partitions_nonconnected(
                self._adj_list, elements=self._contracted_indices
            )
        else:
            partitions = get_all_partitions(self._contracted_indices)
        for partition in partitions:
            weight, subexpression = self._get_subexpression(partition)
            yield weight, subexpression

    def _get_subexpression(self, partition: List[Set[str]]) -> Tuple[float, str]:
        weight = partition_weight(partition)
        mapping = {}
        subexpression = self.expression
        for part in partition:
            rep = min(part)
            for index in part:
                if (
                    self._reserved_indices is not None
                    and index not in self._reserved_indices
                ):
                    mapping[index] = rep
        for index in mapping:
            subexpression = subexpression.replace(index, mapping.get(index, index))
        return weight, subexpression

    def calculate(
        self,
        tensors: List[np.ndarray],
        average: bool = True,
        _dediag: bool = True,
        **kwargs,
    ) -> float | np.ndarray:
        backend = get_backend()
        if _dediag:
            if self._reserved_indices:
                warnings.warn(
                    "Dediagonalization is not supported",
                    "for U statistics with tensor result.",
                )
                _dediag = False
            else:
                tensors = backend.dediag_tensors(
                    tensors, sample_size=tensors[0].shape[0]
                )
        result = None
        subexpressions = self._get_all_subexpressions(dediag=_dediag)
        for weight, subexpression in subexpressions:
            if result is None:
                result = backend.einsum(subexpression, *tensors, **kwargs)
            result += weight * backend.einsum(subexpression, *tensors, **kwargs)
        if average:
            i, j = self.input_index_position
            ns = tensors[i].shape[j]
            order = self.order
            return result / backend.prod(range(ns, ns - order, -1))
        return backend.to_numpy(result)

    def __call__(self, *args, **kwds):
        return self.calculate(*args, **kwds)

    def complexity(
        self, optimize: str = "greedy", n: int = 10**4, _dediag: bool = True, **kwargs
    ) -> int:
        """
        Calculate the complexity of the U statistics expression.
        The complexity is defined as the number of multiplications required
        to compute the U statistics.
        """
        if self._reserved_indices:
            raise ValueError(
                "Complexity calculation is not supported for U statistics with reserved indices."
            )
        shapes = [(n,) * len(inputs) for inputs in self._inputs]
        info = ComplexityInfo()
        subexpressions = self._get_all_subexpressions(dediag=_dediag)
        for _, subexpression in subexpressions:
            _, path_info = oe.contract_path(
                subexpression, *shapes, optimize=optimize, shapes=True, **kwargs
            )
            scaling = max(path_info.scale_list)
            flops = path_info.opt_cost
            largest_intermediate = path_info.largest_intermediate
            info.update(scaling, flops, largest_intermediate)
        return info.scaling, info.flops, info.largest_intermediate


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
