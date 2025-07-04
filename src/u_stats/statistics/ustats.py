"""
U-Statistics Computation Module
==============================

This module provides efficient computation of U-statistics using tensor operations
and Einstein summation notation. U-statistics are a fundamental class of statistics
that generalize sample means to functions of multiple observations.

Classes:
    UStats: Main class for tensor-based U-statistics computation
    U_stats_loop: Reference implementation using explicit loops

The module supports:
- Arbitrary order U-statistics
- Multiple input expression formats
- Efficient tensor-based computation
- Computational complexity analysis
"""

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
    "u_stats_loop",
]


class UStats:
    """
    Efficient computation of U-statistics using Einstein summation notation.

    U-statistics are unbiased estimators that generalize sample means to functions
    of multiple observations. This class provides an efficient tensor-based
    implementation that avoids explicit loops and theoretically supports arbitrary
    order U-statistics with flexible expression formats.

    Parameters
    ----------
    expression : str or tuple or list
        The Einstein summation expression defining the U-statistic structure,
        which define the decomposition form of the U-statistic's kernel

        Supported formats:
        - **String**: Einstein notation (e.g., 'ij,jk->', 'ab,bc,ca->')
        - **Tuple**: (inputs, outputs) where:
            - inputs: sequence of sequences of hashable indices
            - outputs: sequence of hashable output indices
        - **List**: input indices only (output defaults to scalar)

    Attributes
    ----------
    expression : str
        The Einstein summation expression used for computation.
    order : int
        The order of the U-statistic.

    Examples
    --------
    Create and compute a second-order U-statistic:

    >>> import numpy as np
    >>> from u_stats import UStats
    >>>
    >>> # String notation
    >>> ustat = UStats('ij,ji->')
    >>> tensor1 = np.random.randn(100, 100)
    >>> tensor2 = np.random.randn(100, 100)
    >>> result = ustat.compute([tensor1, tensor2])
    >>> print(f"U-statistic: {result:.4f}")

    Using tuple notation for more complex expressions:

    >>> # Tuple notation: ([[i,j], [j,k], [k,i]], [])
    >>> ustat_cycle = UStats(([['i','j'], ['j','k'], ['k','i']], []))
    >>>
    >>> # List notation (scalar output)
    >>> ustat_simple = UStats([['i','j'], ['j','i']])

    Third-order U-statistic example:

    >>> # Three-way interaction U-statistic
    >>> ustat3 = UStats('abc,bca,cab->')
    >>> tensors = [np.random.randn(50, 50, 50) for _ in range(3)]
    >>> result = ustat3.compute(tensors)

    Notes
    -----
    The implementation uses several key optimizations:

    1. **Tensor Decomposition**: Decomposes U-statistics into weighted
       subexpressions based on index partitions
    2. **Backend Support**: Works with NumPy and PyTorch tensors
    3. **Optimization**: Leverages opt_einsum for efficient contraction paths
    """

    def __init__(self, expression: str | Tuple[Inputs, Outputs] | Inputs):
        # Parse expression into internal format
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
        """
        The Einstein summation expression of the U-statistic's
        decomposition form.

        Returns
        -------
        str
            Einstein summation expression in string format (e.g., 'ij,ji->').
        """
        return self._ep

    @cached_property
    def _adj_list(self) -> Dict[str, Set[str]]:
        """Adjacency list representation of index connections."""
        return get_adj_list(self._inputs)

    @cached_property
    def order(self) -> int:
        """
        The order of the U-statistic.

        The order equals the number of contracted (summed) indices, which
        corresponds to the number of sample observations in each term.

        Returns
        -------
        int
            Order of the U-statistic (≥ 1).

        Examples
        --------
        >>> ustat = UStats('ij,ji->')  # 2 contracted indices: i, j
        >>> print(ustat.order)  # Output: 2
        >>>
        >>> ustat3 = UStats('abc,bca,cab->')  # 3 contracted indices: a, b, c
        >>> print(ustat3.order)  # Output: 3
        """
        return len(self._contracted_indices)

    @cached_property
    def input_index_position(self) -> Tuple[int, int]:
        """Position of the first contracted index in input tensors."""
        for i, inputs in enumerate(self._inputs):
            for index in self._contracted_indices:
                pos = inputs.find(index)
                if pos != -1:
                    return (i, pos)

    def get_all_subexpressions(
        self, dediag: bool = True
    ) -> Generator[Tuple[float, str], None, None]:
        """
        Generate all subexpressions and weights for U-statistic computation.

        This method decomposes the U-statistic into a sum of V-statistics
        (subexpressions) with appropriate weights using index partitions.

        Parameters
        ----------
        dediag : bool, default=True
            Whether to use dediagonalization (exclude diagonal terms).
            When True, only considers non-connected partitions.

        Yields
        ------
        tuple of (float, str)
            Weight and Einstein summation subexpression for each partition.

        Notes
        -----
        The decomposition follows the principle that U-statistics can be
        expressed as linear combinations of V-statistics with specific
        weights determined by the partition structure.
        """
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
        """
        Generate a subexpression and weight from an index partition.

        Parameters
        ----------
        partition : list of sets
            A partition of the contracted indices into disjoint subsets.

        Returns
        -------
        tuple of (float, str)
            The partition weight and corresponding Einstein summation expression.
        """
        weight = partition_weight(partition)
        mapping = {}
        subexpression = self.expression

        # Create index mapping for each partition
        for part in partition:
            rep = min(part)  # Representative index for this partition
            for index in part:
                if (
                    self._reserved_indices is not None
                    and index not in self._reserved_indices
                ):
                    mapping[index] = rep

        # Apply mapping to create subexpression
        for index in mapping:
            subexpression = subexpression.replace(index, mapping.get(index, index))

        return weight, subexpression

    def compute(
        self,
        tensors: List[np.ndarray],
        average: bool = True,
        _dediag: bool = True,
        **kwargs,
    ) -> float | np.ndarray:
        """
        Compute the U-statistic on input tensors.

        Parameters
        ----------
        tensors : list of np.ndarray or torch.Tensor
            Input tensors for the V-statistic computation. Each tensor represents
            the tensorization of factors in the V-statistic's kernel. For example,
            if the kernel h = h_1 * h_2 * ... * h_K and each h_k is defined on
            X^d, then T^{(k)}_{i1,i2,...,id} = h_k(X_{i1}, X_{i2}, ..., X_{id}).
        average : bool, default=True
            Whether to return the averaged U-statistic. If False, returns
            the unscaled sum over all valid index combinations.
        _dediag : bool, default=True
            Whether to apply dediagonalization (exclude diagonal terms).
            Automatically disabled for tensor-valued U-statistics.
            Setting to False computes V-statistics instead.
        **kwargs
            Additional keyword arguments passed to opt_einsum.contract().
            Common options include 'optimize' for path optimization strategy.
            ""optimize" can be 'greedy', 'optimal', 'dp', "auto", etc.

        Returns
        -------
        float or np.ndarray
            The computed U-statistic value:
            - **float**: For scalar U-statistics (no output indices)
            - **np.ndarray**: For tensor-valued U-statistics (with output indices)
                              (under testing)
        """
        backend = get_backend()

        tensors = [backend.to_tensor(tensor) for tensor in tensors]
        # Handle dediagonalization
        if _dediag:
            if self._reserved_indices:
                warnings.warn(
                    "Dediagonalization is not supported for U-statistics "
                    "with tensor-valued outputs. Computing V-statistic instead.",
                    UserWarning,
                )
                _dediag = False
            else:
                tensors = backend.dediag_tensors(
                    tensors, sample_size=tensors[0].shape[0]
                )

        # Compute weighted sum of subexpressions
        result = None
        subexpressions = self.get_all_subexpressions(dediag=_dediag)

        for weight, subexpression in subexpressions:
            subresult = backend.einsum(subexpression, *tensors, **kwargs)
            if result is None:
                result = weight * subresult
            else:
                result += weight * subresult

        # Apply normalization if averaging
        if average:
            i, j = self.input_index_position
            sample_size = tensors[i].shape[j]
            order = self.order
            # Normalize by number of k-permutations: n(n-1)...(n-k+1)
            normalization = backend.prod(range(sample_size, sample_size - order, -1))
            result = result / normalization

        return backend.to_numpy(result)

    def __call__(self, *args, **kwargs):
        """
        Compute the U-statistic from input tensors.

        Parameters
        ----------
        tensors : list of np.ndarray or torch.Tensor
            Input tensors for the V-statistic computation. Each tensor represents
            the tensorization of factors in the V-statistic's kernel. For example,
            if the kernel h = h_1 * h_2 * ... * h_K and each h_k is defined on
            X^d, then T^{(k)}_{i1,i2,...,id} = h_k(X_{i1}, X_{i2}, ..., X_{id}).
        average : bool, default=True
            Whether to return the averaged U-statistic. If False, returns
            the unscaled sum over all valid index combinations.
        _dediag : bool, default=True
            Whether to apply dediagonalization (exclude diagonal terms).
            Automatically disabled for tensor-valued U-statistics.
            Setting to False computes V-statistics instead.
        **kwargs
            Additional keyword arguments passed to opt_einsum.contract().
            Common options include 'optimize' for path optimization strategy.
            ""optimize" can be 'greedy', 'optimal', 'dp', "auto", etc.

        Returns
        -------
        float or np.ndarray
            The computed U-statistic value:
            - **float**: For scalar U-statistics (no output indices)
            - **np.ndarray**: For tensor-valued U-statistics (with output indices)
            (under testing)

        Raises
        ------
        ValueError
            If tensors have incompatible shapes or if the sample size is too small
            for the requested U-statistic order.
        """
        return self.compute(*args, **kwargs)

    def complexity(
        self, optimize: str = "greedy", n: int = 10**3, _dediag: bool = True, **kwargs
    ) -> Tuple[int, float, int]:
        """
        Analyze the computational complexity of the U-statistic expression.

        Estimates the computational cost in terms of scaling exponent,
        floating-point operations, and memory requirements.

        Parameters
        ----------
        optimize : str, default="greedy"
            Optimization strategy for Einstein summation path finding.
            Options include 'greedy', 'optimal', 'dp', 'branch-2', etc.
            Same as opt_einsum.contract_path().
        n : int, default=1000
            Sample size to use for complexity analysis.
        _dediag : bool, default=True
            Whether to include dediagonalization in the analysis.
        **kwargs
            Additional arguments passed to opt_einsum.contract_path().

        Returns
        -------
        tuple of (int, float, int)
            Complexity metrics:
            - **scaling**: Computational scaling exponent (max over subexpressions)
            - **flops**: Total floating-point operations count
            - **largest_intermediate**: Size of largest intermediate tensor

        Raises
        ------
        ValueError
            If complexity analysis is requested for tensor-valued U-statistics
            (those with output indices), which is not currently supported.

        Examples
        --------
        >>> ustat = UStats('ij,ji->')
        >>> scaling, flops, memory = ustat.complexity(n=1000)
        >>> print(f"Scaling: O(n^{scaling})")
        >>> print(f"FLOPs: {flops:.2e}")
        >>> print(f"Memory: {memory} elements")

        Compare optimization strategies:

        >>> # Greedy optimization
        >>> s1, f1, m1 = ustat.complexity(optimize='greedy')
        >>> # Optimal optimization
        >>> s2, f2, m2 = ustat.complexity(optimize='optimal')
        >>> print(f"Greedy FLOPs: {f1:.2e}, Optimal FLOPs: {f2:.2e}")

        Notes
        -----
        The complexity analysis:
        - Considers all subexpressions in the U-statistic decomposition
        - Reports the worst-case metrics across all subexpressions
        - Helps in choosing appropriate optimization strategies
        """
        if self._reserved_indices:
            raise ValueError(
                "Complexity analysis is not supported for U-statistics "
                "with output indices (tensor-valued results)."
            )

        # Create dummy tensor shapes for analysis
        shapes = [(n,) * len(inputs) for inputs in self._inputs]
        info = ComplexityInfo()

        # Analyze each subexpression
        subexpressions = self.get_all_subexpressions(dediag=_dediag)
        for _, subexpression in subexpressions:
            _, path_info = oe.contract_path(
                subexpression, *shapes, optimize=optimize, shapes=True, **kwargs
            )
            scaling = max(path_info.scale_list)
            flops = path_info.opt_cost
            largest_intermediate = path_info.largest_intermediate
            info.update(scaling, flops, largest_intermediate)

        return info.scaling, info.flops, info.largest_intermediate


def u_stats_loop(tensors: List[np.ndarray], expression: List[List[int]] | str) -> float:
    """
    Compute U-statistics using explicit loops (reference implementation).

    This function provides a straightforward but computationally expensive
    implementation using explicit loops over all permutations of sample indices.
    It serves as a reference for correctness verification and educational purposes.

    Parameters
    ----------
    tensors : list of np.ndarray
        Input tensors for the U-statistic computation. All tensors must have
        the same size along the sample dimension (first axis).
    expression : list of lists of int
        Index specification for each tensor. Each inner list contains the
        indices used by the corresponding tensor. For example:
        - [[0, 1], [1, 0]] corresponds to 'ij,ji->'
        - [[0, 1, 2], [1, 2, 0]] corresponds to 'abc,bca->'

    Returns
    -------
    float
        The computed U-statistic value (always scalar).

    Warnings
    --------
    This implementation has O(n^k) time complexity where n is the sample size
    and k is the U-statistic order. For sample sizes > 100 or orders > 3,
    use the UStats class instead for practical computation.

    Examples
    --------
    Second-order U-statistic:

    >>> import numpy as np
    >>> X = np.random.randn(20, 5)
    >>> Y = np.random.randn(5, 20)
    >>> expression = [[0, 1], [1, 0]]  # Corresponds to 'ij,ji->'
    >>> result = U_stats_loop([X, Y], expression)
    >>> print(f"Loop-based result: {result:.6f}")

    Compare with tensor-based method:

    >>> ustat = UStats('ij,ji->')
    >>> tensor_result = ustat.compute([X, Y])
    >>> print(f"Tensor-based result: {tensor_result:.6f}")
    >>> print(f"Difference: {abs(result - tensor_result):.2e}")

    Third-order example (small sample size recommended):

    >>> tensors = [np.random.randn(10, 3, 3) for _ in range(3)]
    >>> expression = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    >>> result = U_stats_loop(tensors, expression)

    Notes
    -----
    **Algorithm**: For each permutation of k distinct sample indices:

    1. Extract the corresponding tensor elements using the expression
    2. Compute their product
    3. Sum over all valid permutations
    4. Normalize by the number of k-permutations: n(n-1)...(n-k+1)

    **Use Cases**:
    - Verification of tensor-based implementations
    - Educational demonstrations of U-statistic concepts
    - Small-scale computations where simplicity is preferred over efficiency
    - Debugging complex tensor expressions

    **Limitations**:
    - Exponential time complexity makes it impractical for large samples
    - Only supports scalar-valued U-statistics
    - No backend support or GPU acceleration
    """
    if isinstance(expression, str):
        expression, _ = einsum_eq_to_strlist(expression)
    num_tensors = len(tensors)
    sample_size = tensors[0].shape[0]
    expression = standardize_indices(expression)
    order = len(set(itertools.chain(*expression)))

    # Compute normalization factor
    num_permutations = np.prod(np.arange(sample_size, sample_size - order, -1))
    total_sum = 0.0

    # Loop over all k-permutations of sample indices
    for indices in itertools.permutations(range(sample_size), order):
        product = 1.0

        # Compute product over all tensors
        for tensor_idx in range(num_tensors):
            # Extract indices for this tensor according to expression
            current_indices = tuple(indices[j] for j in expression[tensor_idx])
            product *= tensors[tensor_idx][current_indices]

        total_sum += product

    return total_sum / num_permutations
