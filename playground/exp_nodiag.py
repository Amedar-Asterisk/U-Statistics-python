from U_stats.utils import get_adj_list
from U_stats.statistics.U2V import (
    get_all_partitions_nonconnected,
    get_all_partitions,
    bell_number,
)
from typing import List, Tuple, Union


def test_mode(m: int) -> List[Union[int, Tuple[int, int]]]:
    """Generate mode configuration for U-statistics testing.

    Args:
        m: Number of tensors.

    Returns:
        List of modes where first and last elements are single integers,
        and middle elements are pairs of consecutive integers.
    """
    outputs = []
    for i in range(m + 1):
        if i == 0:
            outputs.append([i])
        elif i == m:
            outputs.append([i - 1])
        else:
            outputs.append([i - 1, i])
    return outputs


def test1(m: int):
    mode = test_mode(m)
    adj_list = get_adj_list(mode)
    number_partitions_nonconnected = len(
        list(get_all_partitions_nonconnected(adj_list))
    )
    number_partitions_all = bell_number(m)
    print(f"number_partitions_all: {number_partitions_all}")
    print(f"number_partitions_nonconnected: {number_partitions_nonconnected}")
    print(f"rate: {number_partitions_nonconnected / number_partitions_all}")


def test2(m: int):
    mode = test_mode(m)
    adj_list = get_adj_list(mode)
    partitions = get_all_partitions_nonconnected(adj_list)
    print(mode)
    for partition in partitions:
        print(partition)


if __name__ == "__main__":
    m = 4
    test1(m)
