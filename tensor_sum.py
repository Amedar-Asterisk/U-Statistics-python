import string
import numpy as np

AB_list = list(string.ascii_lowercase)


def normalize_indexes(lst):
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
    compute_format, result_pair_list = compute_format(index_list)
    return np.einsum(compute_format, *tensor_list), result_pair_list


if __name__ == "__main__":
    lst = [(3, 5), (4, 3, 5), (2, 3, 1)]
    normalized_lst, normalized_dict = normalize_indexes(lst)
    print("标准化后的列表:", normalized_lst)
    print("求和式：", compute_format(lst)[0])
    print("对应的原始指标：", compute_format(lst)[1])
