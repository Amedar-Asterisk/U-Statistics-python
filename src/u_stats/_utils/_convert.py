from typing import Optional, Tuple, List
from ._alphabet import ALPHABET
from ._typing import NestedHashableSequence


def numbers_to_letters(numbers: NestedHashableSequence) -> List[str]:
    try:
        result = []
        letter_to_number = {}

        for lst in numbers:
            if isinstance(lst, int):
                letter = ALPHABET[lst]
                result.append(letter)
                letter_to_number[letter] = lst
            elif isinstance(lst, list | tuple):
                combined = ""
                for num in lst:
                    letter = ALPHABET[num]
                    combined += letter
                    letter_to_number[letter] = num
                result.append(combined)
            else:
                raise TypeError(
                    "Input must be a list of integers or list of integer lists"
                )
        return result
    except IndexError as e:
        raise ValueError(e)


def standardize_indices(expression: NestedHashableSequence) -> List[List[int]]:
    num_to_index = {}
    standardized_lst = []
    current_index = 0
    for t in expression:
        standardized_pair = []
        for num in t:
            if num not in num_to_index:
                num_to_index[num] = current_index
                current_index += 1
            standardized_pair.append(num_to_index[num])
        standardized_lst.append(standardized_pair)
    return standardized_lst


def strlist_to_einsum_eq(inputs: List[List[str]], output: Optional[str] = None) -> str:
    if output is None:
        output = ""
    inputs = [input if isinstance(input, str) else "".join(input) for input in inputs]
    return "->".join([",".join(inputs), output])


def einsum_eq_to_strlist(expression: str) -> Tuple[List[str], str]:
    """Convert an Einstein summation expression to a mode string.

    Args:
        expression (str): The Einstein summation expression (e.g., 'ij,jk->ik').

    Returns:
        Tuple[str, str]: A tuple containing the left-hand side
            and right-hand side modes.
    """
    input, output = expression.split("->")
    inputs = input.split(",")
    return inputs, output
