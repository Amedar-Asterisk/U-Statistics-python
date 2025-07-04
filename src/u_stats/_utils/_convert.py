"""
Conversion utilities for U-Statistics computations.

This module provides utility functions for converting between different representations
used in U-statistics calculations, including number-to-letter mappings, index
standardization, and Einstein summation notation string manipulations.
"""

from typing import Optional, Tuple, List
from ._alphabet import ALPHABET
from ._typing import NestedHashableSequence


def numbers_to_letters(numbers: List[List[int]]) -> List[str]:
    """
    Convert a list of integer lists/integers to corresponding letter representations.

    This function maps integers to letters using a predefined alphabet. It handles
    both individual integers and lists/tuples of integers, combining them into
    string representations.

    Args:
        numbers: A list containing either integers or lists/tuples of integers.
                Each integer is mapped to a corresponding letter from the alphabet.

    Returns:
        A list of strings where each string represents the letter mapping of the
        corresponding input element.

    Raises:
        ValueError: If any integer in the input exceeds the alphabet size.
        TypeError: If input contains elements that are neither integers nor
                  lists/tuples of integers.

    Examples:
        >>> numbers_to_letters([0, [1, 2], (3, 4)])
        ['a', 'bc', 'de']

        >>> numbers_to_letters([[0, 1], [2, 3]])
        ['ab', 'cd']
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
        return result
    except IndexError as e:
        raise ValueError(e)


def standardize_indices(expression: NestedHashableSequence) -> List[List[int]]:
    """
    Standardize indices in a nested sequence to consecutive integers starting from 0.

    This function takes a nested sequence where inner elements can be any hashable
    type and maps them to standardized integer indices. Each unique element gets
    a unique consecutive integer index starting from 0, preserving the order of
    first appearance.

    Args:
        expression: A nested sequence where each inner sequence contains hashable
                   elements that need to be standardized to integer indices.

    Returns:
        A list of lists where each inner list contains the standardized integer
        indices corresponding to the original elements.

    Examples:
        >>> standardize_indices([('a', 'b'), ('b', 'c'), ('a', 'c')])
        [[0, 1], [1, 2], [0, 2]]

        >>> standardize_indices([[1, 3], [3, 5], [1, 5]])
        [[0, 1], [1, 2], [0, 2]]
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


def strlist_to_einsum_eq(inputs: List[List[str]], output: Optional[str] = None) -> str:
    """
    Convert a list of string inputs and optional output to Einstein summation notation.

    This function creates an Einstein summation (einsum) equation string from a list
    of input index specifications and an optional output specification. Each input
    can be either a string or a list of strings that will be joined.

    Args:
        inputs: A list where each element is either a string or a list of strings
               representing the indices for each input tensor.
        output: Optional string specifying the output indices. If None, defaults
               to an empty string.

    Returns:
        A string in Einstein summation notation format: "input1,input2,...->output"

    Examples:
        >>> strlist_to_einsum_eq([['i', 'j'], ['j', 'k']], 'ik')
        'ij,jk->ik'

        >>> strlist_to_einsum_eq(['ab', 'bc'], 'ac')
        'ab,bc->ac'

        >>> strlist_to_einsum_eq(['ii'])
        'ii->'
    """
    if output is None:
        output = ""
    inputs = [input if isinstance(input, str) else "".join(input) for input in inputs]
    return "->".join([",".join(inputs), output])


def einsum_eq_to_strlist(expression: str) -> Tuple[List[str], str]:
    """
    Parse an Einstein summation equation string into input and output components.

    This function takes an Einstein summation notation string and splits it into
    its constituent parts: a list of input index strings and the output index string.

    Args:
        expression: A string in Einstein summation notation format, e.g., "ij,jk->ik"

    Returns:
        A tuple containing:
        - List of input index strings (one for each input tensor)
        - Output index string

    Examples:
        >>> einsum_eq_to_strlist('ij,jk->ik')
        (['ij', 'jk'], 'ik')

        >>> einsum_eq_to_strlist('abc,bcd,cde->ae')
        (['abc', 'bcd', 'cde'], 'ae')

        >>> einsum_eq_to_strlist('ii->')
        (['ii'], '')
    """
    input, output = expression.split("->")
    inputs = input.split(",")
    return inputs, output
