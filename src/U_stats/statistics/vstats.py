from typing import List, Hashable, Generator, Tuple, Dict, Set, Sequence
from functools import cached_property
from .._utils import (
    standardize_indices,
    get_backend,
    numbers_to_letters,
    strlist_to_einsum_eq,
    einsum_eq_to_strlist,
    Inputs,
    Outputs,
)
import numpy as np
import itertools

__all__ = ["VStats"]


class VStats:
    """A class for calculating the U statistics of a list of kernel
    matrices(tensors) with particular expression."""

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
                output = None
                inputs = numbers_to_letters(standardize_indices(inputs))
            self._ep = strlist_to_einsum_eq(inputs, output)
        self._reserved_indices = output
        self._inputs = inputs
        self._contracted_indices = set(itertools.chain(*inputs))
        self._contracted_indices.discard(self._reserved_indices)

    @property
    def expression(self) -> str:
        """Get the expression of the U statistics."""
        return self._ep

    @cached_property
    def order(self) -> int:
        """Get the order of the U statistics."""
        return len(self._contracted_indices)

    @cached_property
    def output_indices_positions(self) -> Dict[str, Tuple[int, int]]:
        if self._reserved_indices is None:
            return None
        positions = []
        for index in self._reserved_indices:
            for i, input_indices in enumerate(self._inputs):
                pos = input_indices.find(index)
                if pos != -1:
                    positions.append((i, pos))
                    break
        return positions

    @cached_property
    def input_index_position(self) -> Tuple[int, int]:
        for i, inputs in enumerate(self._inputs):
            for index in self._contracted_indices:
                pos = inputs.find(index)
                if pos != -1:
                    return (i, pos)

    def calculate(
        self, tensors: List[np.ndarray], average: bool = True, **kwargs
    ) -> float | np.ndarray:
        """Calculate the U statistics for the given tensors."""
        backend = get_backend()
        result = backend.einsum(self._ep, *tensors, **kwargs)

        if average:
            i, j = self.input_index_position
            ns = tensors[i].shape[j]
            order = self.order
            return result / backend.prod(range(ns, ns - order, -1))
        return result
