from typing import Optional, Tuple, List
from ._typing import _StrExpression, _IntExpression, Expression
from .alphabet import ALPHABET


def numbers_to_letters(numbers: _IntExpression) -> Tuple[List[str], dict]:
    """Convert numbers or lists of numbers to corresponding letters or letter
    combinations, and return a mapping from letters back to numbers.

    Args:
        numbers (List[int] | List[List[int]]):
            List of integers or list of integer lists
            Each integer is converted to corresponding
            lowercase letter (0->a, 1->b, etc)

    Returns:
        Tuple[List[str], dict]: A tuple containing:
            - List of converted letters or letter combinations
            - Dictionary mapping letters back to their original numbers

    Raises:
        TypeError: If input is not a list of integers or list of integer lists
        ValueError: If any number exceeds alphabet size (26)

    Example:
        >>> letters, mapping = numbers_to_letters([0, 1, 2])
        >>> letters
        ['a', 'b', 'c']
        >>> mapping
        {'a': 0, 'b': 1, 'c': 2}
        >>> letters, mapping = numbers_to_letters([[0, 1], [2, 3]])
        >>> letters
        ['ab', 'cd']
        >>> mapping
        {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    """
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
        return result, letter_to_number
    except IndexError as e:
        raise ValueError(e)


def standardize_indices(expression: Expression) -> _IntExpression:
    """Standardize the index list to continuous integer indexes.

    Args:
        lst (list): List of index pairs. Each pair is a list of some integers.

    Returns:
        List[List[int]]: Standardized index list.
    """
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


def expression_to_einsum_equation(
    inputs: _StrExpression, output: Optional[str] = None
) -> str:
    """Convert left-hand side list and right-hand side string to arrow format
    notation.

    Args:
        lhs_lst (list): List of strings representing left-hand side operands
        rhs (str, optional): Right-hand side result string. Defaults to empty string

    Returns:
        str: Formatted string in the form "op1,op2,...->result"

    Example:
        >>> strings2format(['ab', 'bc'], 'ac')
        'ab,bc->ac'
        >>> strings2format(['ab', 'bc'])
        'ab,bc->'
    """
    if output is None:
        output = ""
    return "->".join([",".join(inputs), output])


def einsum_equation_to_expression(expression: str) -> Tuple[_StrExpression, str]:
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
