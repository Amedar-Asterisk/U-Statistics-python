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


def get_all_partitions_nonconnected(
    adj_list: Dict[int, Set[Hashable]], elements: Union[int, Sequence[T]] = None
) -> Generator[List[Set[Union[int, T]]], None, None]:
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


def encoding_func(expression: List[List[int]]) -> Callable[[List[Set[int]]], List[str]]:
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
    len_lst = [len(s) for s in partition]
    num_partitions = len(len_lst)
    sign = (-1) ** (sum(len_lst) - num_partitions)
    value = np.prod([factorial(m - 1) for m in len_lst])
    return sign * value
