"""
V-Statistics Computation Module
===============================

This module provides efficient computation of V-statistics using tensor operations
and Einstein summation notation.

Classes:
    VStats: Main class for tensor-based V-statistics computation

The module supports:
- Arbitrary order V-statistics
- Efficient tensor-based computation
"""

from typing import List, Tuple, Dict
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
    """
    Efficient computation of V-statistics using Einstein summation notation.

    V-statistics are biased estimators that generalize sample means to functions
    of multiple observations. Unlike U-statistics, V-statistics is much
    computationally simpler then U-statistics but potentially biased estimators.

    Parameters
    ----------
    expression : str or tuple or list
        The Einstein summation expression defining the V-statistic structure,
        which define the decomposition form of the V-statistic's kernel

        Supported formats:
        - **String**: Einstein notation (e.g., 'ij,jk->k', 'ab,bc,ca->')
        - **Tuple**: (inputs, outputs) where:
            - inputs: sequence of sequences of hashable indices
            - outputs: sequence of hashable output indices
        - **List**: input indices only (output defaults to scalar)

    Attributes
    ----------
    expression : str
        The Einstein summation expression used for computation.
    order : int
        The order of the V-statistic.

    Examples
    --------
    Create and compute a second-order V-statistic:

    >>> import numpy as np
    >>> from u_stats import VStats
    >>>
    >>> # String notation
    >>> vstat = VStats('ij,ji->')
    >>> tensor1 = np.random.randn(100, 100)
    >>> tensor2 = np.random.randn(100, 100)
    >>> result = vstat.compute([tensor1, tensor2])
    >>> print(f"V-statistic: {result:.4f}")

    Using tuple notation for tensor-valued outputs:

    >>> # Tuple notation with output indices: ([[i,j], [j,k]], [k])
    >>> vstat_tensor = VStats(([['i','j'], ['j','k']], ['k']))
    >>> result = vstat_tensor.compute([tensor1, tensor2])
    >>> print(f"Tensor result shape: {result.shape}")

    Third-order V-statistic example:

    >>> # Three-way interaction V-statistic
    >>> vstat3 = VStats('abc,bca,cab->')
    >>> tensors = [np.random.randn(50, 50, 50) for _ in range(3)]
    >>> result = vstat3.compute(tensors)

    Notes
    -----
    The key differences from U-statistics:

    1. **No Dediagonalization**: V-statistics include all terms, including
       diagonal elements where sample indices may repeat
    2. **Biased Estimation**: Generally biased estimators but simpler to compute
    3. **Tensor Outputs**: Naturally supports tensor-valued outputs without
       restrictions
    4. **Direct Computation**: Uses direct Einstein summation without
       decomposition into subexpressions
    """

    def __init__(self, expression: str | Tuple[Inputs, Outputs] | Inputs):
        """
        Initialize a V-statistic with the given expression.

        Parameters
        ----------
        expression : str or tuple or list
            The Einstein summation expression defining the V-statistic structure.

            Supported formats:
            - **String**: Einstein notation (e.g., 'ij,jk->k', 'ab,bc,ca->')
            - **Tuple**: (inputs, outputs) where:
                - inputs: sequence of sequences of hashable indices
                - outputs: sequence of hashable output indices
            - **List**: input indices only (output defaults to scalar)

        Examples
        --------
        String notation:
        >>> vstat = VStats('ij,ji->')  # Matrix trace-like operation

        Tuple notation with outputs:
        >>> vstat = VStats(([['i','j'], ['j','k']], ['k']))  # Matrix multiplication

        List notation (scalar output):
        >>> vstat = VStats([['i','j'], ['j','i']])  # Scalar result
        """
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
        """
        The Einstein summation expression of the V-statistic.

        Returns
        -------
        str
            Einstein summation expression in string format (e.g., 'ij,ji->').
        """
        return self._ep

    @cached_property
    def order(self) -> int:
        """
        The order of the V-statistic.

        The order equals the number of contracted (summed) indices, which
        corresponds to the number of sample observations in each term.

        Returns
        -------
        int
            Order of the V-statistic (≥ 1).

        Examples
        --------
        >>> vstat = VStats('ij,ji->')  # 2 contracted indices: i, j
        >>> print(vstat.order)  # Output: 2
        >>>
        >>> vstat3 = VStats('abc,bca,cab->')  # 3 contracted indices: a, b, c
        >>> print(vstat3.order)  # Output: 3
        """
        return len(self._contracted_indices)

    @cached_property
    def output_indices_positions(self) -> Dict[str, Tuple[int, int]]:
        """
        Get positions of output indices in input tensors.

        For tensor-valued V-statistics, this method returns the positions
        of each output index within the input tensor list.

        Returns
        -------
        dict or None
            Dictionary mapping output indices to their (tensor_index, axis_position)
            in the input tensors. Returns None for scalar V-statistics.

        Examples
        --------
        >>> vstat = VStats('ij,jk->k')  # Output index 'k'
        >>> positions = vstat.output_indices_positions
        >>> print(positions)  # {'k': (1, 1)} - index 'k' at tensor 1, position 1
        """
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
        """
        Position of the first contracted index in input tensors.

        This method finds the position of any contracted index in the input
        tensors, which is used for determining sample size during normalization.

        Returns
        -------
        tuple of (int, int)
            (tensor_index, axis_position) of the first contracted index found.

        Examples
        --------
        >>> vstat = VStats('ij,ji->')
        >>> pos = vstat.input_index_position
        >>> print(pos)  # (0, 0) - first contracted index at tensor 0, position 0
        """
        for i, inputs in enumerate(self._inputs):
            for index in self._contracted_indices:
                pos = inputs.find(index)
                if pos != -1:
                    return (i, pos)

    def compute(
        self, tensors: List[np.ndarray], average: bool = True, **kwargs
    ) -> float | np.ndarray:
        """
        Compute the V-statistic on input tensors.

        This method performs direct Einstein summation to compute the V-statistic
        without any dediagonalization step, making it simpler but potentially
        biased compared to U-statistics.

        Parameters
        ----------
        tensors : list of np.ndarray or torch.Tensor
            Input tensors for the V-statistic computation. Each tensor represents
            the tensorization of factors in the V-statistic's kernel. For example,
            if the kernel h = h_1 * h_2 * ... * h_K and each h_k is defined on
            X^d, then T^{(k)}_{i1,i2,...,id} = h_k(X_{i1}, X_{i2}, ..., X_{id}).
        average : bool, default=True
            Whether to return the averaged V-statistic. If False, returns
            the unscaled sum over all valid index combinations.
        **kwargs
            Additional keyword arguments passed to the backend's einsum function.
            Common options include 'optimize' for path optimization strategy.

        Returns
        -------
        float or np.ndarray
            The computed V-statistic value:
            - **float**: For scalar V-statistics (no output indices)
            - **np.ndarray**: For tensor-valued V-statistics (with output indices)

        Examples
        --------
        Scalar V-statistic:
        >>> import numpy as np
        >>> vstat = VStats('ij,ji->')
        >>> X = np.random.randn(100, 50)
        >>> Y = np.random.randn(50, 100)
        >>> result = vstat.compute([X, Y])
        >>> print(f"V-statistic: {result:.4f}")

        Tensor-valued V-statistic:
        >>> vstat_tensor = VStats('ij,jk->k')
        >>> X = np.random.randn(100, 50)
        >>> Y = np.random.randn(50, 20)
        >>> result = vstat_tensor.compute([X, Y])
        >>> print(f"Result shape: {result.shape}")  # (20,)

        Without averaging (raw sum):
        >>> raw_result = vstat.compute([X, Y], average=False)
        >>> print(f"Raw sum: {raw_result:.2e}")

        Notes
        -----
        The V-statistic computation follows these steps:

        1. **Direct Summation**: Apply Einstein summation directly to input tensors
        3. **Normalization**: If average=True, divide by n^k where n is sample size
           and k is the order

        **Key Differences from U-statistics**:
        - No decomposition into subexpressions
        - No exclusion of diagonal terms
        - Generally biased but computationally simpler
        - Naturally supports tensor-valued outputs
        """
        backend = get_backend()
        result = backend.einsum(self._ep, *tensors, **kwargs)

        if average:
            i, j = self.input_index_position
            ns = tensors[i].shape[j]
            order = self.order
            return result / backend.prod(range(ns, ns - order, -1))
        return backend.to_numpy(result)

    def __call__(self, *args, **kwargs):
        """
        Compute the V-statistic from input tensors.

        This is a convenience method that calls compute() with the same arguments.

        Parameters
        ----------
        tensors : list of np.ndarray or torch.Tensor
            Input tensors for the V-statistic computation. Each tensor represents
            the tensorization of factors in the V-statistic's kernel. For example,
            if the kernel h = h_1 * h_2 * ... * h_K and each h_k is defined on
            X^d, then T^{(k)}_{i1,i2,...,id} = h_k(X_{i1}, X_{i2}, ..., X_{id}).
        average : bool, default=True
            Whether to return the averaged V-statistic.
        **kwargs
            Additional keyword arguments passed to the backend's einsum function.

        Returns
        -------
        float or np.ndarray
            The computed V-statistic value.

        See Also
        --------
        compute : The main computation method with detailed documentation.
        """
        return self.compute(*args, **kwargs)
