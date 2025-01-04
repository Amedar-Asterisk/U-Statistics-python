### some useful functions
from collections import defaultdict
import string, re
import numpy as np

AB_talbe = list(string.ascii_lowercase)


# A decorator that can make a single argument of a single value function support list
def support_list(func):
    def wrapper(
        is_lst_args=False,
        *args,
    ):
        if not is_lst_args:
            return func(*args)
        else:
            return [func(*args) for args in is_lst_args]

    return wrapper


# This function is to generate the reverse mapping of a dictionary
def reverse_mapping(mapping: dict) -> dict:
    reverse = defaultdict(list)
    for key, value in mapping.items():
        reverse[value].append(key)
    return reverse


# This functio is to standardize the indexes to continuous integer indexes
def standardize_indexes(lst: list) -> tuple:
    """
    Standardize the index list to continuous integer indexes.

    Args:
        lst (list): List of index pairs. Each pair is a list of some integers.

    Returns:
        tuple: Standardized index list and mapping from standardized to original indexes.
    """
    num_to_index = {}
    standardized_lst = []
    standardized_to_original = {}
    current_index = 0
    for t in lst:
        standardized_pair = []
        for num in t:
            if num not in num_to_index:
                num_to_index[num] = current_index
                standardized_to_original[current_index] = num
                current_index += 1
            standardized_pair.append(num_to_index[num])
        standardized_lst.append(standardized_pair)

    return standardized_lst, standardized_to_original


def numbers_to_letters(numbers: list[int] | list[list]) -> list:
    if isinstance(numbers[0], int):
        return [AB_talbe[num] for num in numbers]
    elif isinstance(numbers[0], list):
        return ["".join([AB_talbe[num] for num in pair]) for pair in numbers]
    else:
        raise TypeError(
            "Input must be a list of integers or a list of lists of integers."
        )


def dedup(strings: str | list) -> str:
    """
    Remove the duplicated characters from a string or a list of strings.

    Args:
        strings (str | list): The input string or a list of strings.

    Returns:
        str: A string without duplicated characters.
             If the input is a list, the result is the concatenation of all strings after deduplication.

    Examples:
        >>> dedup("hello")
        "helo"
        >>> dedup(["hello", "world"])
        "hellowrd"
        >>> dedup("")
        ""
        >>> dedup([])
        ""
    """
    if isinstance(strings, str):
        return "".join(dict.fromkeys(strings))
    elif isinstance(strings, list):
        combined_string = "".join(strings)
        return "".join(dict.fromkeys(combined_string))
    else:
        raise TypeError("Input must be a string or a list of strings.")


def strings2format(lhs_lst: list, rhs: str) -> str:
    return "->".join([",".join(lhs_lst), rhs])


def analyze_path(compute_format: str, optimize=False, tensor_dim=10):
    lhs, result = compute_format.split("->")
    lhs_lst = lhs.split(",")
    compute_format = strings2format(lhs_lst, result)
    ndim_lst = [len(pair) for pair in lhs_lst]
    tensor_list = [np.random.rand(*(tensor_dim,) * ndim) for ndim in ndim_lst]
    path_info = np.einsum_path(compute_format, *tensor_list, optimize=optimize)[1]
    naive_flop_match = re.search(r"Naive FLOP count:\s*([\d\.]+e[+-]?\d*)", path_info)
    optimized_flop_match = re.search(
        r"Optimized FLOP count:\s*([\d\.]+e[+-]?\d*)", path_info
    )
    largest_intermediate_match = re.search(
        r"Largest intermediate:\s*([\d\.]+e[+-]?\d*)", path_info
    )

    naive_flop = naive_flop_match.group(1) if naive_flop_match else None
    optimized_flop = optimized_flop_match.group(1) if optimized_flop_match else None
    largest_intermediate = (
        largest_intermediate_match.group(1) if largest_intermediate_match else None
    )

    return float(naive_flop), float(optimized_flop), float(largest_intermediate)
