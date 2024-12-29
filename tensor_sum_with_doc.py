"""张量求和运算模块。

此模块提供了对张量进行标准化索引处理和爱因斯坦求和约定计算的功能。
主要用于处理多维张量的运算，支持动态索引映射和格式化。
"""

import string
import numpy as np

AB_list = list(string.ascii_lowercase)


def normalize_indexes(lst):
    """标准化索引列表，将任意数字索引映射到连续的整数索引。

    Parameters
    ----------
    lst : list of tuple
        原始索引列表，每个元素是包含数字的元组

    Returns
    -------
    tuple
        normalized_lst : list
            标准化后的索引列表
        index_to_num : dict
            标准化索引到原始数字的映射字典

    Examples
    --------
    >>> lst = [(3, 5), (4, 3, 5)]
    >>> normalized_lst, mapping = normalize_indexes(lst)
    >>> print(normalized_lst)
    [(0, 1), (2, 0, 1)]
    """
    num_to_index = {}
    normalized_lst = []
    index_to_num = {}
    current_index = 0
    for t in lst:
        normalized_tuple = []
        for num in t:
            if num not in num_to_index:
                num_to_index[num] = current_index
                index_to_num[current_index] = num
                current_index += 1
            normalized_tuple.append(num_to_index[num])
        normalized_lst.append(tuple(normalized_tuple))

    return normalized_lst, index_to_num


def compute_format(index_list: list):
    """计算爱因斯坦求和约定的格式字符串。

    Parameters
    ----------
    index_list : list
        索引列表，每个元素是表示张量维度的元组

    Returns
    -------
    tuple
        format_str : str
            爱因斯坦求和约定格式字符串
        original_indices : tuple
            结果张量对应的原始索引

    Examples
    --------
    >>> lst = [(3, 5), (4, 3, 5)]
    >>> format_str, indices = compute_format(lst)
    >>> print(format_str)
    'ab,bac->abc'
    """
    normalized_indexes, normalizing_mapping = normalize_indexes(index_list)
    ab_index_map = {AB_list[i]: j for i, j in normalizing_mapping.items()}

    sum_parts = []
    all_letters = set()

    for pair in normalized_indexes:
        current_str = "".join(AB_list[k] for k in pair)
        sum_parts.append(current_str)
        for k in pair:
            all_letters.add(AB_list[k])
    result_format = "".join(sorted(all_letters))
    compute_format = "->".join([",".join(sum_parts), result_format])
    return compute_format, tuple([ab_index_map[letter] for letter in result_format])


def tensor_sum(tensor_list: list, index_list: list):
    """执行张量求和运算。

    使用爱因斯坦求和约定计算多个张量的缩并。

    Parameters
    ----------
    tensor_list : list
        张量列表，每个元素是一个numpy数组
    index_list : list
        索引列表，描述每个张量的维度对应关系

    Returns
    -------
    tuple
        result : ndarray
            求和结果
        indices : tuple
            结果张量的维度索引

    Examples
    --------
    >>> tensors = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    >>> indices = [(0, 1), (1, 2)]
    >>> result, final_indices = tensor_sum(tensors, indices)
    """
    compute_format, result_pair_list = compute_format(index_list)
    return np.einsum(compute_format, *tensor_list), result_pair_list


if __name__ == "__main__":
    lst = [(3, 5), (4, 3, 5), (2, 3, 1)]
    normalized_lst, normalized_dict = normalize_indexes(lst)
    print("标准化后的列表:", normalized_lst)
    print("求和式：", compute_format(lst)[0])
    print("对应的原始指标：", compute_format(lst)[1])
