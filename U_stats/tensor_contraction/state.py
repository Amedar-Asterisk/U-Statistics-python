from ..utils import reverse_mapping, dedup, strings2format
from .mode import StandardizedMode
from typing import List, Tuple, Dict, Set, Union, Hashable
import math
from copy import deepcopy
from dataclasses import dataclass


@dataclass
class IndexRegistry:
    """
    An efficient index manager for tensor network computations.

    This class implements fast index-to-location mapping using dictionaries and sets.
    It's primarily used for maintaining relationships between indices and tensor positions

    Attributes:
        _location_map (Dict[str, Set[int]]): Maps indices to their positions
            - key: index string
            - value: set of positions where the index appears

    Example:
        ```python
        # Create a new registry
        registry = IndexRegistry()

        # Add some index mappings
        registry.append('i', 0)  # Add index 'i' at position 0
        registry.append('i', 1)  # Add index 'i' at position 1

        # Get all positions for index 'i'
        positions = registry.locations('i')  # Returns {0, 1}

        # Remove an index mapping
        registry.remove('i', 0)  # Remove index 'i' from position 0
        ```
    """

    __slots__ = ["_location_map"]

    def __init__(self):
        """
        Initialize an empty index registry.
        """
        self._location_map: Dict[str, Set[int]] = {}

    def copy(self):
        """
        Create a deep copy of the index registry.

        Returns:
            IndexRegistry: A new registry object with the same index mappings
        """
        new_registry = IndexRegistry()
        new_registry._location_map = deepcopy(self._location_map)
        return new_registry

    @property
    def indexes(self) -> Set[str]:
        """
        Get all registered unique indices.

        Returns:
            Set[str]: A set of all unique indices currently in the registry

        Time Complexity: O(1)
        """
        return set(self._location_map.keys())

    def append(self, index: str, location: int) -> None:
        """
        Add a new index-position mapping.

        Args:
            index (str): The index to add
            location (int): The position where this index appears

        Note:
            If the index already exists, the new position will be added to its set
            of positions. If the position already exists for this index, it won't
            be added again.
        """
        if index not in self._location_map:
            self._location_map[index] = set()
        self._location_map[index].add(location)

    def remove(self, index: str, location: int) -> None:
        """
        Remove an index-position mapping.

        Args:
            index (str): The index to remove
            location (int): The position to remove for this index

        Note:
            - Uses set.discard() instead of remove() to avoid KeyError when position
              doesn't exist
            - Removes the index completely if it has no more positions
        """
        if index in self._location_map:
            self._location_map[index].discard(location)
            if not self._location_map[index]:
                del self._location_map[index]

    def merge(self, str_set: List[str], objective: str) -> None:
        if objective not in self._location_map:
            self._location_map[objective] = set()
        for c in str_set:
            if c in self._location_map:
                positions = self._location_map.pop(c)
                self._location_map[objective].update(positions)

    def locations(self, index: str) -> Set[int]:
        """
        Get all positions where an index appears.

        Args:
            index (str): The index to query

        Returns:
            Set[int]: Set of positions where the index appears.
                     Returns empty set if index doesn't exist.

        """
        return self._location_map.get(index, set())


class TensorContractionState(StandardizedMode):
    """
    A class that manages the state of tensor contractions in a tensor network.

    This class handles the bookkeeping of tensor indices during contraction operations,
    including tracking index pairs, managing their positions, and computing contraction paths.

    Attributeself._data (Dict[int, str]): Maps position to index pairs
        _index_table (IndexRegistry): Registry tracking index locations
    """

    def __init__(self, mode: List[Union[str, List[Hashable]]]):
        """
        Initialize a tensor contraction state.

        Args:
            index_pairs (list[str]): List of index pairs for each tensor. Each pair is
                                    represented as a string of indices.

        Example:
            >>> state = TensorContractionState(['ij', 'jk', 'kl'])
        """
        super().__init__(mode)
        pair_dict = dict.fromkeys(range(len(mode)), None)
        for i, pair in enumerate(mode):
            if len(pair) > 0:
                pair_dict[i] = pair
        self._data: Dict[int, str] = pair_dict
        self._index_table = IndexRegistry()

        for i, pair in self._data.items():
            for index in pair:
                self._index_table.append(index, i)

    def copy(self):
        """
        Create a deep copy of the state.

        Returns:
            TensorContractionState: A new state object with the same index pairs
        """
        new_state = TensorContractionState([])
        new_state._data = self._data.copy()
        new_state._index_table = self._index_table.copy()
        return new_state

    def __len__(self) -> int:
        """
        Get the number of tensor pairs in the state.

        Returns:
            int: Number of tensor pairs
        """
        return len(self._data)

    def __getitem__(self, position: int) -> str:
        """
        Get the index pair at a specific position.

        Args:
            position (int): Position of the desired pair

        Returns:
            str: The index pair at the specified position
        """
        return self.pair(position)

    @property
    def pair_dict(self) -> Dict[int, str]:
        """
        Get the sequence of index pairs as an immutable tuple.

        Returns:
            tuple[str]: Tuple of index pairs in order.
        """
        return self._data.copy()

    @property
    def indexes(self) -> set[str]:
        """
        Get all unique indices in the tensor network.

        Returns:
            set[str]: Tuple of unique indices.
        """
        return set(self._index_table.indexes)

    def neighbours(self, index: str, pair_position: int) -> str:
        """
        Get the neighboring indices of a given index in a specific pair.

        Args:
            index (str): The target index
            pair_position (int): Position of the pair containing the index

        Returns:
            str: String of neighboring indices with target index removed
        """
        return self.pair(pair_position).replace(index, "")

    def pair(self, position: int) -> str:
        """
        Get the index pair at a specific position.

        Args:
            position (int): Position of the desired pair

        Returns:
            str: The index pair at the specified position
        """
        return self._data[position] if position in self._data else ""

    def times(self, index: str) -> int:
        """
        Count how many times an index appears in the tensor network.

        Args:
            index (str): The index to count

        Returns:
            int: Number of occurrences of the index
        """
        return len(self._index_table.locations(index))

    def replace(self, positions: list[int] | int, pair: str) -> int:
        """
        Replace multiple index pairs with a single new pair.

        Args:
            positions (list[int] | int): Positions or one position of pairs to replace
            pair (str): New index pair to insert

        Returns:
            int: Position where the new pair was inserted
        """
        positions_set = set(positions)
        frontmost = min(positions_set)

        for pos in positions_set:
            for idx in self.pair(pos):
                self._index_table.remove(idx, pos)

        self._data[frontmost] = pair

        for idx in pair:
            self._index_table.append(idx, frontmost)

        if pair is not None:
            positions_set.remove(frontmost)

        for pos in positions_set:
            self._data.pop(pos)

        if self._data[0] == "":
            self._data.pop(0)

        return frontmost

    def dedup(self) -> Dict[int, str]:
        """
        Remove duplicate indices within each tensor pair while preserving order.

        This method processes each tensor pair in the state and removes any duplicate
        indices that appear within the same pair. It maintains the original order of
        first appearance for each index.

        Returns:
            Dict[int, str]: A dictionary mapping positions to string representations
                of format changes ('old->new'). Only includes positions where
                duplicates were removed.

        Example:
            If a tensor pair at position 0 is 'iii', it will be converted to 'i'
            and the return dict will contain {0: 'iii->i'}

        Time Complexity: O(n * k) where n is number of pairs and k is max length of any pair
        Space Complexity: O(k) where k is max length of any pair
        """
        compute_format = {}
        for k in range(len(self)):
            # Use ordered dictionary to maintain index order while removing duplicates
            seen = {}
            result = ""
            pair = self.pair(k)

            # Process each index, keeping only first occurrence
            for idx in pair:
                if idx not in seen:
                    seen[idx] = True
                    result += idx

            # Only store changes where duplicates were removed
            if len(result) < len(pair):
                compute_format[k] = f"{pair}->{result}"
                self._data[k] = result

        return compute_format

    def contract(self, index: str) -> Tuple[List[int], int, str]:
        """
        Contract a specific index in the tensor network.

        This method performs the contraction operation by:
        1. Finding all positions where the index appears
        2. Computing the resulting index pair after contraction
        3. Updating the state to reflect the contraction

        Args:
            index (str): The index to contract

        Returns:
            Tuple[List[int], int, str]: A tuple containing:
                - List of positions involved in contraction
                - Position where result is stored
                - String representation of the contraction operation
        """
        positions = self._index_table.locations(index)
        compute_positions = sorted(positions)

        pairs = [self.pair(pos) for pos in compute_positions]
        result = dedup(pairs).replace(index, "")
        compute_format = strings2format(
            [self.pair(pos) for pos in compute_positions], result
        )
        save_position = self.replace(compute_positions, result)

        return compute_positions, save_position, compute_format

    def eval_result_length(self, index: str = None) -> Union[None, int]:
        """
        Evaluate the length of resulting tensor after contracting an index.

        This method calculates the number of unique indices that would remain
        after contracting a given index.

        Args:
            index (str, optional): The index to evaluate. If None, evaluates all indices.

        Returns:
            Union[None, int]:
                - If index is None: Updates length_indicator and returns None
                - If index provided: Returns the length of resulting tensor

        Time Complexity:
            - Single index: O(p) where p is number of positions containing the index
            - All indices: O(n*p) where n is number of unique indices
        """
        if index is None:
            index_length = {}
            for idx in self.indexes:
                length = self.eval_result_length(idx)
                index_length[idx] = length

            self.length_indicator = reverse_mapping(index_length)
            return None

        positions = self._index_table.locations(index)

        unique_indices = set()
        for pos in positions:
            pair = self.pair(pos)
            for idx in pair:
                if idx != index:
                    unique_indices.add(idx)

        return len(unique_indices)

    def next_contract(self) -> str:
        """
        Determine the next index to contract based on optimization criteria.

        The method uses two criteria:
        1. Choose indices that result in smallest intermediate tensors
        2. If multiple such indices exist, choose the one appearing least times

        Returns:
            str: The index chosen for next contraction
        """
        self.eval_result_length()
        indexes_min_length = self.length_indicator[min(self.length_indicator)]
        if len(indexes_min_length) == 1:
            return indexes_min_length[0]
        else:
            return min(indexes_min_length, key=lambda x: self.times(x))


def critical_complexity_order(k: int) -> int:
    return k * (k - 1) // 2 if k % 2 != 0 else k * (k - 1) // 2 + k // 2


def complexity_order(m: int) -> tuple[int, int]:
    if m <= 0:
        return (1, 2)
    k = int((1 + math.sqrt(1 + 8 * m)) / 2) + 1

    left, right = 1, k
    while left < right:
        mid = (left + right) // 2
        if critical_complexity_order(mid) <= m:
            left = mid + 1
        else:
            right = mid

    return left - 1
