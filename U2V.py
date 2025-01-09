from typing import List, Set
from math import factorial
import numpy as np


def partition(m: int, k: int) -> List[List[Set[int]]]:
    """
    Partition the numbers [0, 1, ..., m-1] into k non-empty subsets.

    Args:
        m (int): The total number of elements (from 0 to m-1).
        k (int): The number of subsets to partition into (k < m).

    Returns:
        List[List[Set[int]]]: A list of all possible partitions. Each partition is a list of k subsets,
                              where each subset is represented as a set of integers.

    Example:
        >>> partition(4, 2)
        [
            [{0, 1, 2}, {3}],
            [{0, 1, 3}, {2}],
            [{0, 1}, {2, 3}],
            [{0, 2, 3}, {1}],
            [{0, 2}, {1, 3}],
            [{0, 3}, {1, 2}],
            [{0}, {1, 2, 3}]
        ]
    """

    def backtrack(
        pos: int, current_partition: List[Set[int]], result: List[List[Set[int]]]
    ) -> None:
        """
        Recursively generate all possible partitions.

        Args:
            pos (int): The current number being processed.
            current_partition (List[Set[int]]): The current partition being constructed.
            result (List[List[Set[int]]]): The list to store all valid partitions.
        """
        # If all numbers have been assigned
        if pos == m:
            # Ensure all k subsets are non-empty
            if all(len(subset) > 0 for subset in current_partition):
                result.append([subset.copy() for subset in current_partition])
            return

        # Calculate the number of remaining numbers and subsets
        remaining_numbers = m - pos
        remaining_subsets = k - len(current_partition)

        # If remaining numbers equal remaining subsets, each number must go into a new subset
        if remaining_numbers == remaining_subsets:
            for num in range(pos, m):
                current_partition.append(set([num]))
            backtrack(m, current_partition, result)
            for _ in range(remaining_subsets):
                current_partition.pop()
            return

        # Try to place the current number into an existing subset
        for subset in current_partition:
            subset.add(pos)
            backtrack(pos + 1, current_partition, result)
            subset.remove(pos)

        # Try to place the current number into a new subset
        if len(current_partition) < k:
            current_partition.append(set([pos]))
            backtrack(pos + 1, current_partition, result)
            current_partition.pop()

    result: List[List[Set[int]]] = []
    backtrack(0, [], result)
    return result


def encode_partition(partition: List[Set[int]]) -> List[int]:
    """
    Encode a partition using the minimum value of each subset.

    Args:
        partition (List[Set[int]]): A partition of numbers, where each subset is represented as a set of integers.

    Returns:
        List[int]: A list of length m, where each element represents the minimum value of the subset containing that number.

    Example:
        >>> encode_partition([{0, 1, 2}, {3}])
        [0, 0, 0, 3]
    """
    # Create a mapping from each number to the minimum value of its subset
    num_to_min = {}
    for subset in partition:
        min_val = min(subset)
        for num in subset:
            num_to_min[num] = min_val

    # Determine the maximum number to know the size of the encoded list
    max_num = max(num_to_min.keys(), default=-1)

    # Build the encoded list
    encoded = []
    for num in range(max_num + 1):
        if num in num_to_min:
            encoded.append(num_to_min[num])
        else:
            # Handle numbers not in any subset (if necessary)
            encoded.append(-1)  # or raise an error, depending on requirements

    return encoded


def partition_weights(partition: List[Set[int]]) -> int:
    len_lst = [len(s) for s in partition]
    num_partitions = len(len_lst)
    sign = (-1) ** (sum(len_lst) - num_partitions)
    value = np.prod([factorial(n - 1) for n in len_lst])
    return sign * value
