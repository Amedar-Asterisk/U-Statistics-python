"""
Conversion Utilities for U-Statistics Computation
=================================================

This module provides essential utility functions for converting between different
representations used in U-statistics calculations. It handles the complex mappings
between various index notations, enabling flexible specification of U-statistic
expressions while maintaining computational efficiency.

The module supports conversions between:
- Integer indices and alphabetic letter representations
- Nested sequences and standardized integer indices
- List-based specifications and Einstein summation notation strings
- Einstein notation parsing and component extraction

Functions:
    numbers_to_letters: Convert integer indices to alphabetic representations
    standardize_indices: Normalize nested sequences to consecutive integer indices
    strlist_to_einsum_eq: Build Einstein summation notation from string lists
    einsum_eq_to_strlist: Parse Einstein notation into component strings

These utilities are fundamental to the U-statistics framework, enabling users to
specify complex tensor operations using intuitive notation while internally
converting to optimized computational representations.

Examples
--------
Basic index conversion workflow:

>>> from u_stats._utils._convert import numbers_to_letters, standardize_indices
>>>
>>> # Start with symbolic indices
>>> expression = [['i', 'j'], ['j', 'k'], ['k', 'i']]
>>>
>>> # Standardize to integers
>>> standardized = standardize_indices(expression)
>>> print(standardized)  # [[0, 1], [1, 2], [2, 0]]
>>>
>>> # Convert to letters
>>> letters = numbers_to_letters(standardized)
>>> print(letters)  # ['ab', 'bc', 'ca']

Einstein notation conversion:

>>> from u_stats._utils._convert import strlist_to_einsum_eq, einsum_eq_to_strlist
>>>
>>> # Build Einstein equation
>>> inputs = ['ij', 'jk', 'ki']
>>> equation = strlist_to_einsum_eq(inputs, '')
>>> print(equation)  # 'ij,jk,ki->'
>>>
>>> # Parse back to components
>>> parsed_inputs, output = einsum_eq_to_strlist(equation)
>>> print(parsed_inputs)  # ['ij', 'jk', 'ki']
>>> print(output)  # ''

Complex U-statistic specification:

>>> # Third-order U-statistic with cycle structure
>>> cycle_indices = [['a', 'b', 'c'], ['b', 'c', 'a'], ['c', 'a', 'b']]
>>> standardized = standardize_indices(cycle_indices)
>>> letters = numbers_to_letters(standardized)
>>> equation = strlist_to_einsum_eq(letters)
>>> print(equation)  # 'abc,bca,cab->'

Notes
-----
**Index Mapping**: The conversion system maintains a consistent mapping between
different representations, ensuring that the same logical index structure is
preserved across all formats.

**Alphabet Limitations**: The current implementation supports up to 26 distinct
indices (limited by the English alphabet). For larger index sets, consider
extending the alphabet or using purely numerical representations.

**Einstein Notation**: The module follows standard Einstein summation convention
where repeated indices are summed over, and the arrow notation (->) separates
inputs from outputs.
"""

from typing import Optional, Tuple, List
from ._alphabet import ALPHABET
from ._typing import NestedHashableSequence


def numbers_to_letters(numbers: List[List[int]]) -> List[str]:
    """
    Convert integer indices to corresponding alphabetic letter representations.

    This function maps integers to letters using a predefined alphabet, enabling
    conversion from numeric index specifications to readable string representations.
    It handles both individual integers and sequences of integers, combining them
    into concatenated string representations.

    Parameters
    ----------
    numbers : list of int or list of list of int
        A list containing either integers or lists/tuples of integers.
        Each integer is mapped to a corresponding letter from the alphabet
        (0='a', 1='b', 2='c', etc.).

    Returns
    -------
    list of str
        A list of strings where each string represents the letter mapping
        of the corresponding input element:
        - **Single integers**: Mapped to single letters
        - **Integer sequences**: Mapped to concatenated letter strings

    Raises
    ------
    ValueError
        If any integer in the input exceeds the alphabet size (≥26).
    TypeError
        If input contains elements that are neither integers nor
        lists/tuples of integers.

    Examples
    --------
    Mixed integer and sequence inputs:

    >>> numbers_to_letters([0, [1, 2], (3, 4)])
    ['a', 'bc', 'de']

    Multiple sequences:

    >>> numbers_to_letters([[0, 1], [2, 3]])
    ['ab', 'cd']

    Single integers only:

    >>> numbers_to_letters([0, 1, 2])
    ['a', 'b', 'c']

    Complex U-statistic index mapping:

    >>> # Third-order U-statistic indices
    >>> indices = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    >>> letters = numbers_to_letters(indices)
    >>> print(letters)  # ['abc', 'bca', 'cab']

    Notes
    -----
    **Alphabet Limitation**: The function supports up to 26 distinct indices,
    corresponding to the English alphabet. Attempting to use indices ≥26 will
    raise a ValueError.

    **Order Preservation**: The mapping preserves the order of indices within
    each sequence and the order of sequences in the input list.

    **Use Case**: This function is primarily used in converting standardized
    integer indices to readable Einstein summation notation components.
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
    first appearance across all sequences.

    Parameters
    ----------
    expression : NestedHashableSequence
        A nested sequence where each inner sequence contains hashable elements
        that need to be standardized to integer indices. Inner sequences can
        be lists, tuples, or any sequence type containing hashable elements.

    Returns
    -------
    list of list of int
        A list of lists where each inner list contains the standardized integer
        indices corresponding to the original elements. The mapping maintains:
        - **Uniqueness**: Each distinct element gets a unique integer
        - **Consistency**: Same elements map to same integers across sequences
        - **Order**: Integers assigned in order of first appearance

    Examples
    --------
    String-based indices:

    >>> standardize_indices([('a', 'b'), ('b', 'c'), ('a', 'c')])
    [[0, 1], [1, 2], [0, 2]]

    Mixed hashable types:

    >>> standardize_indices([[1, 3], [3, 5], [1, 5]])
    [[0, 1], [1, 2], [0, 2]]

    Complex nested structure:

    >>> standardize_indices([['x', 'y', 'z'], ['y', 'z', 'x'], ['z', 'x', 'y']])
    [[0, 1, 2], [1, 2, 0], [2, 0, 1]]

    U-statistic kernel indices:

    >>> # Second-order U-statistic with symmetric kernel
    >>> kernel_indices = [('i', 'j'), ('j', 'i')]
    >>> standardized = standardize_indices(kernel_indices)
    >>> print(standardized)  # [[0, 1], [1, 0]]

    Higher-order example:

    >>> # Third-order cycle U-statistic
    >>> cycle = [('a', 'b', 'c'), ('b', 'c', 'a'), ('c', 'a', 'b')]
    >>> standardized = standardize_indices(cycle)
    >>> print(standardized)  # [[0, 1, 2], [1, 2, 0], [2, 0, 1]]

    Notes
    -----
    **Index Assignment**: Indices are assigned based on the order of first
    appearance when scanning through the nested sequence from left to right,
    top to bottom.

    **Hashable Requirement**: All elements in the inner sequences must be
    hashable (can be used as dictionary keys). This includes strings, numbers,
    tuples, but excludes lists and dictionaries.

    **Preservation of Structure**: The function preserves the exact structure
    of the input, only converting the leaf elements to standardized integers.

    **Use Case**: This function is essential for converting user-friendly
    symbolic index specifications (using letters, names, etc.) to the
    numeric format required for efficient tensor operations.
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
    Convert string lists to Einstein summation notation equation.

    This function creates an Einstein summation (einsum) equation string from a list
    of input index specifications and an optional output specification. It handles
    both string inputs and lists of strings, automatically joining list elements
    into concatenated index strings.

    Parameters
    ----------
    inputs : list of str or list of list of str
        A list where each element specifies the indices for one input tensor:
        - **str**: Direct index string (e.g., 'ij', 'abc')
        - **list of str**: List of individual indices to be joined (e.g., ['i', 'j'])
    output : str, optional
        String specifying the output indices. If None, defaults to an empty
        string, indicating scalar output.

    Returns
    -------
    str
        Einstein summation equation in the format "input1,input2,...->output".

    Examples
    --------
    Matrix multiplication specification:

    >>> strlist_to_einsum_eq([['i', 'j'], ['j', 'k']], 'ik')
    'ij,jk->ik'

    Direct string inputs:

    >>> strlist_to_einsum_eq(['ab', 'bc'], 'ac')
    'ab,bc->ac'

    Scalar output (trace-like operations):

    >>> strlist_to_einsum_eq(['ii'])
    'ii->'

    Complex multi-tensor contraction:

    >>> inputs = [['a', 'b', 'c'], ['b', 'c', 'd'], ['d', 'a']]
    >>> equation = strlist_to_einsum_eq(inputs, 'a')
    >>> print(equation)  # 'abc,bcd,da->a'

    U-statistic expression building:

    >>> # Second-order U-statistic
    >>> u_inputs = [['i', 'j'], ['j', 'i']]
    >>> u_equation = strlist_to_einsum_eq(u_inputs, '')
    >>> print(u_equation)  # 'ij,ji->'

    Third-order cycle U-statistic:

    >>> cycle_inputs = [['a', 'b', 'c'], ['b', 'c', 'a'], ['c', 'a', 'b']]
    >>> cycle_eq = strlist_to_einsum_eq(cycle_inputs)
    >>> print(cycle_eq)  # 'abc,bca,cab->'

    Notes
    -----
    **Input Flexibility**: The function accepts both pre-joined strings and
    lists of individual index characters, providing flexibility in how users
    specify tensor contractions.

    **Output Convention**: An empty output string indicates a scalar result,
    which is common for U-statistics that compute single numerical values.

    **Index Validation**: The function does not validate index consistency
    (e.g., checking that contracted indices appear exactly twice). Such
    validation is typically performed by the Einstein summation implementation.

    **Use Case**: This function is essential for converting internal list-based
    representations to the string format required by einsum operations.
    """
    if output is None:
        output = ""
    inputs = [input if isinstance(input, str) else "".join(input) for input in inputs]
    return "->".join([",".join(inputs), output])


def einsum_eq_to_strlist(expression: str) -> Tuple[List[str], str]:
    """
    Parse Einstein summation equation string into input and output components.

    This function takes an Einstein summation notation string and decomposes it
    into its constituent parts: a list of input index strings (one for each
    input tensor) and the output index string. This is the inverse operation
    of `strlist_to_einsum_eq`.

    Parameters
    ----------
    expression : str
        A string in Einstein summation notation format with the structure
        "input1,input2,...,inputN->output" where each input represents the
        indices for one tensor and output specifies the result indices.

    Returns
    -------
    tuple of (list of str, str)
        A tuple containing:
        - **List of str**: Input index strings, one for each input tensor
        - **str**: Output index string (empty string for scalar output)

    Examples
    --------
    Matrix multiplication parsing:

    >>> einsum_eq_to_strlist('ij,jk->ik')
    (['ij', 'jk'], 'ik')

    Multi-tensor contraction:

    >>> einsum_eq_to_strlist('abc,bcd,cde->ae')
    (['abc', 'bcd', 'cde'], 'ae')

    Scalar output (U-statistic):

    >>> einsum_eq_to_strlist('ii->')
    (['ii'], '')

    Complex U-statistic expression:

    >>> einsum_eq_to_strlist('ijk,jkl,lij->')
    (['ijk', 'jkl', 'lij'], '')

    Higher-order tensor operations:

    >>> equation = 'abcd,bcde,cdef,defa->af'
    >>> inputs, output = einsum_eq_to_strlist(equation)
    >>> print(f"Inputs: {inputs}")  # ['abcd', 'bcde', 'cdef', 'defa']
    >>> print(f"Output: {output}")  # 'af'

    Trace and diagonal operations:

    >>> # Trace: sum of diagonal elements
    >>> einsum_eq_to_strlist('ii->')
    (['ii'], '')
    >>>
    >>> # Diagonal extraction
    >>> einsum_eq_to_strlist('ij->i')  # Would need ii->i for diagonal
    (['ij'], 'i')

    Notes
    -----
    **Format Requirements**: The input string must follow the exact Einstein
    summation format with a single '->' separator. Malformed strings will
    raise errors during the split operation.

    **Empty Output**: An empty output string (after '->') indicates scalar
    output, which is common in U-statistics computations.

    **Index Consistency**: The function performs simple parsing without
    validating index consistency. Invalid equations may only be detected
    during actual tensor contraction.

    **Use Case**: This function is essential for:
    - Converting stored string representations back to component form
    - Analyzing existing Einstein expressions
    - Building composite operations from existing expressions
    - Debugging and introspection of complex tensor operations
    """
    input, output = expression.split("->")
    inputs = input.split(",")
    return inputs, output
