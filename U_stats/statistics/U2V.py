##################################################
# Description:
# The U2V class is used to convert a U-statistics graph to a V-statistics graph.
# The encode_partition function is used to encode a partition using the minimum value of each subset.
# The partition_weights function is used to calculate the weight of a partition.
# The stirling_number function is used to calculate the Stirling number of the second kind.
# The get_all_partitions function is used to generate all possible partitions of numbers [0, 1, ..., m-1] into k non-empty subsets.
# The partitions function is used to generate all ways to partition numbers [0, 1, ..., m-1] into k non-empty subsets.

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
)
from ..utils import *
from math import factorial
import numpy as np

T = TypeVar("T", bound=Hashable)


@overload
def partitions(m: int, k: int) -> Generator[List[Set[int]], None, None]: ...


@overload
def partitions(
    elements: Union[Sequence[T], Set[T]], k: int
) -> Generator[List[Set[T]], None, None]: ...


def partitions(
    elements: Union[int, Sequence[T], Set[T]], k: int
) -> Generator[List[Set[Union[int, T]]], None, None]:
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
    if isinstance(elements, int):
        m = elements
    else:
        m = len(elements)
    for k in range(1, m + 1):
        yield from partitions(elements, k)


def encoding_func(mode: List[List[int]]) -> Callable[[List[Set[int]]], List[str]]:
    standardized_mode = standardize_indexes(mode)

    def encoded_partition(partition: List[Set[int]]):
        current_index = 0
        mapping = {}
        for s in partition:
            for i in s:
                mapping[i] = AB_table[current_index]
            current_index += 1
        return ["".join([mapping[index] for index in lst]) for lst in standardized_mode]

    return encoded_partition


def partition_weight(partition: List[Set[Hashable]]) -> int:
    len_lst = [len(s) for s in partition]
    num_partitions = len(len_lst)
    sign = (-1) ** (sum(len_lst) - num_partitions)
    value = np.prod([factorial(m - 1) for m in len_lst])
    return sign * value


def stirling_number(m: int, k: int) -> int:
    dp = [[0] * (k + 1) for _ in range(m + 1)]
    dp[0][0] = 1
    for i in range(1, m + 1):
        for j in range(1, k + 1):
            dp[i][j] = j * dp[i - 1][j] + dp[i - 1][j - 1]
    return dp[m][k]
