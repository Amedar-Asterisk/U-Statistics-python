from typing import List, Set, Generator
from math import factorial
import numpy as np


def partitions(m: int, k: int) -> Generator[List[Set[int]], None, None]:
    """
    Optimized version of partition to generate all ways to partition
    numbers [0, 1, ..., m-1] into k non-empty subsets.

    Args:
        m (int): The total number of elements (from 0 to m-1).
        k (int): The number of subsets to partition into (k < m).

    Yields:
        List[Set[int]]: A single partition as a list of k subsets,
                        where each subset is represented as a set of integers.
    """

    def backtrack(
        pos: int, current_partition: List[Set[int]], empty_subsets: int
    ) -> Generator[List[Set[int]], None, None]:
        """
        Recursively generate all possible partitions with optimizations.

        Args:
            pos (int): The current number being processed.
            current_partition (List[Set[int]]): The current partition being constructed.
            empty_subsets (int): The count of empty subsets still available.

        Yields:
            List[Set[int]]: A valid partition.
        """
        # If all elements are processed, yield the partition
        if pos == m:
            if empty_subsets == 0:
                yield [set(subset) for subset in current_partition]
            return

        # Try placing `pos` in each existing subset
        for subset in current_partition:
            subset.add(pos)
            yield from backtrack(pos + 1, current_partition, empty_subsets)
            subset.remove(pos)

        # Try creating a new subset for `pos` if possible
        if empty_subsets > 0:
            current_partition.append({pos})
            yield from backtrack(pos + 1, current_partition, empty_subsets - 1)
            current_partition.pop()

    # Start backtracking with an empty partition and all k subsets available
    yield from backtrack(0, [], k)


def get_all_partitions(m: int) -> Generator[List[Set[int]], None, None]:
    for k in range(1, m + 1):
        yield from partitions(m, k)


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
    value = np.prod([factorial(m - 1) for m in len_lst])
    return sign * value


def stirling_number(m: int, k: int) -> int:
    dp = [[0] * (k + 1) for _ in range(m + 1)]
    dp[0][0] = 1
    for i in range(1, m + 1):
        for j in range(1, k + 1):
            dp[i][j] = j * dp[i - 1][j] + dp[i - 1][j - 1]
    return dp[m][k]
