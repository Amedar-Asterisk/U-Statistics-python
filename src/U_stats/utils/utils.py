from typing import (
    Dict,
    Set,
)
from ._typing import _IntExpression


def get_adj_list(expression: _IntExpression) -> Dict[int, Set[int]]:
    """Convert a mode list to an adjacency list representation.

    Args:
        mode (NestedHashableList): The mode list to convert.

    Returns:
        Dict[int, Set[int]]: The adjacency list representation of the mode.
    """
    vertices = set()
    for pair in expression:
        vertices.update(set(pair))
    adj_list: Dict = dict()
    for index in vertices:
        adj_list[index] = set()
        for pair in expression:
            if index in pair:
                adj_list[index].update(pair)
        adj_list[index].discard(index)
    return adj_list
