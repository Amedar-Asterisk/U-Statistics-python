import einsum_benchmark.instances as ebin
import matplotlib.pyplot as plt
from typing import Tuple


def einsum_expression_to_mode(expression: str) -> Tuple[str, str]:
    """Convert an Einstein summation expression to a mode string.

    Args:
        expression (str): The Einstein summation expression (e.g., 'ij,jk->ik').

    Returns:
        Tuple[str, str]: A tuple containing the left-hand side and right-hand side modes.
    """
    lhs, rhs = expression.split("->")
    lhs_modes = lhs.split(",")
    return lhs_modes, rhs


def discord_instance(instance):
    format_string = instance.format_string
    num_tensors = len(format_string)
    lhs_modes, rhs = einsum_expression_to_mode(format_string)
    len_result = len(rhs)
    indices = set()
    for mode in lhs_modes:
        indices.update(mode)
    num_indices = len(indices)
    return num_tensors, num_indices, len_result


def k_compare(a, b, k):
    return a > k * b


if __name__ == "__main__":
    num = {}
    ks = [1, 2, 3, 5, 10, 20]
    for k in ks:
        num[k] = 0
    for instance in ebin:
        num_tensors, num_indices, len_result = discord_instance(instance)
        for k in ks:
            if k_compare(num_tensors, num_indices, k):
                num[k] += 1
        print(
            f"Instance: {instance.name}, Num Tensors: {num_tensors}, Num Indices: {num_indices}, Result Length: {len_result}"
        )

    print(num)
