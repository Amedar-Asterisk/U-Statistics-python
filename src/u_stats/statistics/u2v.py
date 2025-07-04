"""
U-statistics to V-statistics conversion utilities.

This module provides functions for generating set partitions, encoding partitions,
and computing partition weights. These utilities are essential for converting
U-statistics to V-statistics through the analysis of partition structures.

The main functionality includes:
- Generation of all possible set partitions with specified number of parts
- Generation of non-connected partitions based on adjacency constraints
- Encoding partitions into string representations
- Computing Möbius function weights for partition lattices
"""

from typing import (
    Callable,
    List,
    Set,
    Generator,
    TypeVar,
    Hashable,
    Sequence,
    Union,
    overload,
    Dict,
    MutableSequence,
)
from .._utils import standardize_indices, ALPHABET
from math import factorial
import numpy as np

T = TypeVar("T", bound=Hashable)


def get_adj_list(cover: MutableSequence[MutableSequence[T]]) -> Dict[T, Set[T]]:
    """
    Generate an adjacency list from a cover (collection of subsets).

    This function creates a graph representation where each element is connected
    to all other elements that appear in the same subset within the cover.

    Args:
        cover: A collection of subsets, where each subset contains hashable elements.
               Elements appearing in the same subset will be considered adjacent.

    Returns:
        A dictionary mapping each element to a set of its adjacent elements.
        An element is not considered adjacent to itself.

    Example:
        >>> cover = [['a', 'b'], ['b', 'c'], ['d']]
        >>> get_adj_list(cover)
        {'a': {'b'}, 'b': {'a', 'c'}, 'c': {'b'}, 'd': set()}
    """
    adj_list = {}
    for subset in cover:
        for element in subset:
            if element not in adj_list:
                adj_list[element] = set()
            adj_list[element].update(subset)
            adj_list[element].discard(element)
    return adj_list


@overload
def partitions(  # noqa: E704
    m: int, k: int
) -> Generator[List[Set[int]], None, None]: ...


@overload
def partitions(  # noqa: E704
    elements: Union[Sequence[T], Set[T]], k: int
) -> Generator[List[Set[T]], None, None]: ...


def partitions(
    elements: Union[int, Sequence[T], Set[T]], k: int
) -> Generator[List[Set[Union[int, T]]], None, None]:
    """
    Generate all set partitions of elements into exactly k non-empty parts.

    This function generates all possible ways to partition a set of elements
    into exactly k non-empty subsets. The partitions are generated using
    a backtracking algorithm.

    Args:
        elements: Either an integer n (representing elements 0, 1, ..., n-1),
                 a sequence of elements, or a set of elements to be partitioned.
        k: The number of parts (non-empty subsets) in each partition.

    Yields:
        Lists of sets, where each list represents one partition and each set
        represents one part of the partition.

    Note:
        If k > len(elements), no partitions are generated.

    Example:
        >>> list(partitions(3, 2))
        [[{0}, {1, 2}], [{0, 1}, {2}], [{0, 2}, {1}]]

        >>> list(partitions(['a', 'b', 'c'], 2))
        [[{'a'}, {'b', 'c'}], [{'a', 'b'}, {'c'}], [{'a', 'c'}, {'b'}]]
    """
    if isinstance(elements, int):
        m = elements
        elements = range(elements)
    else:
        elements = list(elements)
        m = len(elements)

    if k > m:
        return

    def backtrack(
        pos: int, current_partition: List[Set[Union[int, T]]], empty_subsets: int
    ) -> Generator[List[Set[Union[int, T]]], None, None]:
        if pos == m:
            if empty_subsets == 0:
                yield [set(subset) for subset in current_partition]
            return

        for subset in current_partition:
            subset.add(elements[pos])
            yield from backtrack(pos + 1, current_partition, empty_subsets)
            subset.remove(elements[pos])

        if empty_subsets > 0:
            current_partition.append({elements[pos]})
            yield from backtrack(pos + 1, current_partition, empty_subsets - 1)
            current_partition.pop()

    yield from backtrack(0, [], k)


def get_all_partitions(
    elements: Union[int, Sequence[T]],
) -> Generator[List[Set[Union[int, T]]], None, None]:
    """
    Generate all possible set partitions of elements.

    This function generates all possible ways to partition a set of elements
    into any number of non-empty parts (from 1 to the total number of elements).

    Args:
        elements: Either an integer n (representing elements 0, 1, ..., n-1)
                 or a sequence of elements to be partitioned.

    Yields:
        Lists of sets, where each list represents one partition and each set
        represents one part of the partition. Partitions are generated in
        order of increasing number of parts.

    Example:
        >>> list(get_all_partitions(3))
        [[{0, 1, 2}], [{0}, {1, 2}], [{0, 1}, {2}], [{0, 2}, {1}], [{0}, {1}, {2}]]
    """
    if isinstance(elements, int):
        m = elements
    else:
        m = len(elements)
    for k in range(1, m + 1):
        yield from partitions(elements, k)


def get_all_partitions_nonconnected(
    adj_list: Dict[int, Set[Hashable]], elements: Union[int, Sequence[T]] = None
) -> Generator[List[Set[Union[int, T]]], None, None]:
    """
    Generate all partitions where no two connected elements are in the same part.

    This function generates partitions with the constraint that if two elements
    are adjacent (connected) in the given adjacency list, they cannot be placed
    in the same partition subset.

    Args:
        adj_list: A dictionary representing the adjacency list of a graph,
                 where keys are vertices and values are sets of adjacent vertices.
        elements: Either an integer n (representing elements 0, 1, ..., n-1),
                 a sequence of elements, or None to use all keys from adj_list.

    Yields:
        Lists of sets representing valid partitions where no two adjacent
        elements appear in the same subset.

    Note:
        This is useful for generating independent set partitions in graph theory
        or for handling constraints in combinatorial optimization problems.

    Example:
        >>> adj_list = {0: {1}, 1: {0, 2}, 2: {1}}  # Path graph 0-1-2
        >>> list(get_all_partitions_nonconnected(adj_list))
        # Will generate partitions where 0 and 1 are separate, 1 and 2 are separate
    """
    if elements is None:
        vertices = list(adj_list.keys())
    else:
        if isinstance(elements, int):
            vertices = list(range(elements))
        else:
            vertices = list(elements)
    graph = adj_list

    def backtrack(
        unassigned: List[Hashable], current_partition: List[Set[Hashable]]
    ) -> Generator[List[Set[Hashable]], None, None]:
        if not unassigned:
            yield [set(part) for part in current_partition]
            return

        vertex = unassigned[0]
        remaining = unassigned[1:]

        for i in range(len(current_partition)):
            part = current_partition[i]
            if not any(neighbor in part for neighbor in graph[vertex]):
                part.add(vertex)
                yield from backtrack(remaining, current_partition)
                part.remove(vertex)

        current_partition.append({vertex})
        yield from backtrack(remaining, current_partition)
        current_partition.pop()

    yield from backtrack(vertices, [])


# abandoned
def encoding_func(expression: List[List[int]]) -> Callable[[List[Set[int]]], List[str]]:
    """
    Create an encoding function that maps partitions to string representations.

    This function creates a closure that can encode any partition into a list
    of strings based on a given expression structure. Each part of the partition
    is assigned a letter from the alphabet, and the expression is encoded using
    these letters.

    Args:
        expression: A list of lists of integers representing the structure
                   to be encoded. The indices should correspond to elements
                   that will appear in partitions.

    Returns:
        A function that takes a partition (list of sets) and returns a list
        of strings where each string represents one subexpression encoded
        according to the partition.

    Example:
        >>> expr = [[0, 1], [1, 2]]
        >>> encoder = encoding_func(expr)
        >>> partition = [{0}, {1, 2}]
        >>> encoder(partition)
        ['ab', 'bb']  # 0 maps to 'a', 1 and 2 map to 'b'

    Note:
        The input expression is automatically standardized to ensure consistent
        indexing starting from 0.
    """
    standardized_expression = standardize_indices(expression)

    def encoded_partition(partition: List[Set[int]]):
        current_index = 0
        mapping = {}
        for s in partition:
            for i in s:
                mapping[i] = ALPHABET[current_index]
            current_index += 1
        return [
            "".join([mapping[index] for index in lst])
            for lst in standardized_expression
        ]

    return encoded_partition


def partition_weight(partition: List[Set[Hashable]]) -> int:
    """
    Calculate the weight of a partition using the Möbius function of partition lattices.

    The weight is computed according to the formula:
    μ(P) = (-1)^(Σ|Si| - k) * Π(|Si| - 1)!

    where:
    - P is the partition
    - |Si| is the size of the i-th subset in the partition
    - k is the number of subsets in the partition
    - Σ denotes sum over all subsets
    - Π denotes product over all subsets

    Args:
        partition: A list of sets representing a partition, where each set
                  is a non-empty subset of the partition.

    Returns:
        The integer weight of the partition according to the Möbius function.
        The sign alternates based on the total number of elements minus
        the number of parts.

    Example:
        >>> partition = [{1, 2}, {3}]
        >>> partition_weight(partition)
        -1  # (-1)^((2+1) - 2) * (2-1)! * (1-1)! = (-1)^1 * 1! * 0! = -1

    Note:
        This weight function is fundamental in the theory of U-statistics
        and their conversion to V-statistics through partition analysis.
    """
    len_lst = [len(s) for s in partition]
    num_partitions = len(len_lst)
    sign = (-1) ** (sum(len_lst) - num_partitions)
    value = np.prod([factorial(m - 1) for m in len_lst])
    return sign * value
