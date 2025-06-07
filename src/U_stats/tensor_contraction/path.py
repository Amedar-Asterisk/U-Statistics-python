from .mode import _StandardizedMode
import warnings  # noqa: F401
from functools import cached_property
from typing import List, Tuple, Dict, Set, Optional, Callable
import itertools
import numpy as np
import opt_einsum as oe
from copy import deepcopy
from dataclasses import dataclass
from ..utils import (
    numbers_to_letters,
    strings2format,
    einsum_expression_to_mode,
    NestedHashableList,
)


@dataclass
class _IndexRegistry:
    __slots__ = ["_location_map"]

    def __init__(self) -> None:
        """Initialize an empty index registry."""
        self._location_map: Dict[int, Set[int]] = {}

    def copy(self) -> "_IndexRegistry":
        """Create a deep copy of the index registry.

        Returns:
            IndexRegistry: A new registry object with the same index mappings
        """
        new_registry = _IndexRegistry()
        new_registry._location_map = deepcopy(self._location_map)
        return new_registry

    @property
    def indices(self) -> Set[int]:
        """Get all registered unique indices.

        Returns:
            Set[str]: A set of all unique indices currently in the registry

        Time Complexity: O(1)
        """
        return set(self._location_map.keys())

    def append(self, index: int, location: int) -> None:
        """Add a new index-position mapping.

        Args:
            index (int): The index to add
            location (int): The position where this index appears

        Note:
            If the index already exists, the new position will be added to its set
            of positions. If the position already exists for this index, it won't
            be added again.
        """
        if index not in self._location_map:
            self._location_map[index] = set()
        self._location_map[index].add(location)

    def remove(self, index: int, location: int) -> None:
        """Remove an index-position mapping.

        Args:
            index (int): The index to remove
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

    def merge(self, int_set: List[int] | int, objective: int) -> None:
        if objective not in self._location_map:
            self._location_map[objective] = set()
        if isinstance(int_set, int):
            int_set = [int_set]
        for c in int_set:
            if c in self._location_map:
                positions = self._location_map.pop(c)
                self._location_map[objective].update(positions)

    def locations(self, index: int) -> List[int]:
        """Get all positions where an index appears.

        Args:
            index (str): The index to query

        Returns:
            Set[int]: Set of positions where the index appears.
                     Returns empty set if index doesn't exist.
        """
        return sorted(list(self._location_map.get(index, set())))


class TensorExpression:

    __size: int = 10**4

    def __init__(self, mode: Optional[NestedHashableList] = None) -> None:
        if mode is not None:
            mode: _StandardizedMode = _StandardizedMode(mode)
            self._position_number = len(mode)
            self.shape = mode.shape
            pair_dict = dict.fromkeys(range(len(mode)), None)
            for i, pair in enumerate(mode._data):
                if len(pair) > 0:
                    pair_dict[i] = pair
            self._pair_dict = pair_dict
            self._index_table: "_IndexRegistry" = _IndexRegistry()
            for i, pair in self._pair_dict.items():
                for index in pair:
                    self._index_table.append(index, i)

        else:
            self._pair_dict = {}
            self._index_table: "_IndexRegistry" = _IndexRegistry()
            self._position_number = 0
            self.shape = None

    def copy(self) -> "TensorExpression":
        new_state = TensorExpression(None)
        new_state._pair_dict = deepcopy(self._pair_dict)
        new_state._index_table = self._index_table.copy()
        new_state.shape = self.shape
        new_state._position_number = self._position_number
        return new_state

    def __str__(self) -> str:
        keys = sorted(self._pair_dict.keys())
        return f"({",".join([str(self._pair_dict[key]) for key in keys])})->"

    def __getitem__(self, index: int) -> List[int]:
        """Get the pair of indices for a given index.

        Args:
            index (int): The index to query

        Returns:
            List[int]: The pair of indices associated with the given index.
                       Returns empty list if index doesn't exist.
        """
        return self._pair_dict[index]

    @cached_property
    def indices(self) -> List[int]:
        return sorted(list(self._index_table.indices))

    @cached_property
    def _indices_mapping(self) -> Dict[int, int]:
        return self._construct_index_mapping(self.indices)

    @cached_property
    def _adj_matrix(self) -> np.ndarray:
        return self._construct_adj_matrix()

    @property
    def index_number(self) -> int:
        return len(self.indices)

    def positions(self, index: int) -> Set[int]:
        return self._index_table.locations(index)

    def _pair_vector(self, pair: List[int]) -> np.ndarray:
        indices_rep = [self._indices_mapping(index) for index in pair]
        return self._bool_vector(set(indices_rep), self.index_number)

    def evaluate(self, path: List[int]) -> int:
        indices = self.indices.copy()
        adj_matrix = self._adj_matrix.copy()
        max_cost = 0
        for index in path:
            if adj_matrix.all():
                max_cost = max(max_cost, len(indices) - 1)
                break
            mapping = self._construct_index_mapping(indices)
            position = mapping(index)
            cost = self._degree(adj_matrix, position)
            adj_matrix = self._eliminate_index_(adj_matrix, position)
            indices.remove(index)
            max_cost = max(max_cost, cost)
        return max_cost

    def exhaustive_search(self) -> Tuple[List[int], int]:
        min_cost = self.index_number
        best_path = None
        for ordering in itertools.permutations(self.indices):
            cost = self.evaluate(ordering)
            if cost < min_cost:
                min_cost = cost
                best_path = ordering
        return best_path, min_cost

    def branch_and_bound_search(self) -> Tuple[List[int], int]:
        if not self.indices:
            return [], 0

        def backtrack(
            indices: List[int],
            path: List[int],
            adj_matrix: np.ndarray,
            current_cost: int,
            index_mapping: callable,
        ) -> None:
            nonlocal min_cost, best_path

            if not indices:
                if current_cost < min_cost:
                    min_cost = current_cost
                    best_path = path.copy()
                return

            if adj_matrix.all():
                remaining_cost = len(indices) - 1
                total_cost = max(current_cost, remaining_cost)
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_path = path + indices
                return

            for i, index in enumerate(indices):
                position = index_mapping(index)
                cost = self._degree(adj_matrix, position)
                max_cost = max(current_cost, cost)

                if max_cost >= min_cost:
                    continue

                old_matrix = adj_matrix.copy()
                new_indices = indices[:i] + indices[i + 1 :]

                adj_matrix = self._eliminate_index_(adj_matrix, position)

                lower_bound = self._degeneracy(adj_matrix)
                if lower_bound > max_cost:
                    adj_matrix = old_matrix
                    continue

                path.append(index)

                backtrack(
                    new_indices,
                    path,
                    adj_matrix,
                    max_cost,
                    self._construct_index_mapping(new_indices),
                )

                path.pop()
                adj_matrix = old_matrix

        best_path, min_cost = self.greedy_search()
        initial_indices = list(self.indices)
        initial_matrix = self._adj_matrix.copy()
        initial_mapping = self._construct_index_mapping(initial_indices)

        backtrack(initial_indices, [], initial_matrix, 0, initial_mapping)
        return best_path, min_cost

    def greedy_search(self) -> Tuple[List[int], int]:
        cost = 0
        path = []
        indices = self.indices.copy()
        adj_matrix = self._adj_matrix.copy()
        index_mapping = self._construct_index_mapping(indices)

        # first pass to remove indices with single position
        for index in indices:
            if len(self.positions(index)) == 1:
                position = index_mapping(index)
                cost = max(cost, self._degree(adj_matrix, position))
                adj_matrix = self._delete_index_(adj_matrix, position)
                indices.remove(index)
                path.append(index)
                index_mapping = self._construct_index_mapping(indices)

        while indices:
            # check if the adjacency matrix is a complete graph
            if adj_matrix.all():
                cost = max(cost, len(indices) - 1)
                path += indices
                break

            # find the indices with the minimum cost
            cost_vector = np.sum(adj_matrix, axis=0) - 1
            min_cost_position = np.argmin(cost_vector)
            min_cost = np.min(cost_vector)
            min_cost_positions = np.where(cost_vector == min_cost)[0]
            # if there are multiple positions with the same cost, choose one randomly
            if len(min_cost_positions) > 1:
                min_cost_position = np.random.choice(min_cost_positions)
            else:
                min_cost_position = min_cost_positions[0]
            min_index = indices[min_cost_position]
            path.append(min_index)
            indices.pop(min_cost_position)
            cost = max(cost, min_cost)
            adj_matrix = self._eliminate_index_(adj_matrix, min_cost_position)

        return path, cost

    def double_greedy_search(self) -> Tuple[List[int], int]:
        cost = 0
        path = []
        indices = self.indices.copy()
        adj_matrix = self._adj_matrix.copy()
        index_mapping = self._construct_index_mapping(indices)

        # first pass to remove indices with single positions
        for index in indices:
            if len(self.positions(index)) == 1:
                position = index_mapping(index)
                cost = max(cost, self._degree(adj_matrix, position))
                adj_matrix = self._delete_index_(adj_matrix, position)
                indices.remove(index)
                path.append(index)
                index_mapping = self._construct_index_mapping(indices)

        while indices:
            # check if the adjacency matrix is a complete graph
            if adj_matrix.all():
                cost = max(cost, len(indices) - 1)
                path += indices
                break

            # find the indices with the minimum cost
            cost_vector = np.sum(adj_matrix, axis=0) - 1
            min_cost_position = np.argmin(cost_vector)
            min_cost = np.min(cost_vector)
            min_cost_positions = np.where(cost_vector == min_cost)[0]

            # find the index with the minimum fill-in from indices with the minimum cost
            if len(min_cost_positions) > 1:
                max_edges = 0
                for position in min_cost_positions:
                    vector = adj_matrix[position]
                    mask = np.outer(vector, vector)
                    edges = np.sum(adj_matrix[mask])
                    if edges > max_edges:
                        max_edges = edges
                        min_cost_position = index
            else:
                min_cost_position = min_cost_positions[0]
            min_index = indices[min_cost_position]
            path.append(min_index)
            indices.pop(min_cost_position)
            cost = max(cost, min_cost)
            adj_matrix = self._eliminate_index_(adj_matrix, min_cost_position)

        return path, cost

    def computing_representation_path(
        self, path: List[int]
    ) -> List[Tuple[List[int], str]]:
        computing_path = []
        registry = self._index_table.copy()
        pair_dict = self._pair_dict.copy()
        position_number = self._position_number
        for index in path:
            positions = registry.locations(index)
            processing_pairs = [pair_dict[i] for i in positions]
            result_pair = self._result_pair(index, processing_pairs)
            computing_format, output_expression = self._pairs_to_format(
                processing_pairs, result_pair=result_pair
            )
            computing_path.append((positions, computing_format))
            if output_expression is not None:
                for index in output_expression:
                    for position in positions:
                        registry.remove(index, position)
                    registry.append(index, position_number)
                pair_dict[position_number] = output_expression
                position_number += 1

            for position in positions:
                pair_dict.pop(position)
        return computing_path

    def path(self, method: str = "greedy") -> Tuple[List[Tuple[List[int], str]], int]:
        if method not in self._METHOD_:
            raise ValueError(
                f"Invalid method: {method}. "
                "Available methods are: {list(self._METHOD_.keys())}"
            )
        index_path, cost = self._METHOD_[method](self)
        computing_path = self.computing_representation_path(index_path)

        return computing_path, cost

    def tupled_path(
        self, method: str = "greedy", analyze: bool = False, optimize: bool = False
    ) -> str:
        path, _ = self.path(method)
        return self.tuplelize_path(
            path,
            optimize=optimize,
            analyze=analyze,
        )

    @staticmethod
    def _construct_index_mapping(indices: List[int]) -> Callable:
        index_dict = {index: i for i, index in enumerate(indices)}

        def mapping(index: int) -> int:
            return index_dict.get(index, -1)

        return mapping

    @staticmethod
    def _bool_vector(set: Set[int], length: int) -> np.ndarray:
        vector = np.zeros(length, dtype=bool)
        vector[list(set)] = True
        return vector

    @staticmethod
    def _result_pair(index: int, pairs: List[List[int]]) -> Set[int]:
        result = set()
        for pair in pairs:
            if index in pair:
                result.update(pair)
        result.discard(index)
        return sorted(list(result))

    @staticmethod
    def _eliminate_index_(adj_matrix: np.ndarray, index: int) -> None:
        vector = adj_matrix[index]
        matrix = np.outer(vector, vector)
        result = adj_matrix | matrix
        result = np.delete(result, index, axis=0)
        result = np.delete(result, index, axis=1)
        return result

    @staticmethod
    def _delete_index_(adj_matrix: np.ndarray, index: int) -> None:
        result = np.delete(adj_matrix, index, axis=0)
        result = np.delete(result, index, axis=1)
        return result

    @staticmethod
    def _degree(adj_matrix: np.ndarray, index: int) -> int:
        vector = adj_matrix[index]
        return np.sum(vector) - 1

    @staticmethod
    def _degeneracy(adj_matrix: np.ndarray) -> int:
        adj_matrix = adj_matrix.copy()
        max_degree = 0
        while True:
            if adj_matrix.all():
                return max(max_degree, adj_matrix.shape[0] - 1)
            vector = np.sum(adj_matrix, axis=0)
            index = np.argmin(vector)
            degree = TensorExpression._degree(adj_matrix, index)
            max_degree = max(max_degree, degree)
            adj_matrix = TensorExpression._delete_index_(adj_matrix, index)

    @staticmethod
    def _pairs_to_format(
        input_pairs: List[Tuple[int]], result_pair: Tuple[int] = None
    ) -> Tuple[str, List[int]]:
        if result_pair is None:
            pairs = input_pairs
            pairs, mapping = numbers_to_letters(pairs)
            return strings2format(pairs[:-1]), None
        else:
            pairs = input_pairs + [result_pair]
            pairs, mapping = numbers_to_letters(pairs)
            return strings2format(pairs[:-1], pairs[-1]), [
                mapping[i] for i in pairs[-1]
            ]

    @staticmethod
    def tuplelize_path(
        computing_path: List[Tuple[List[int], str]],
        optimize: Optional[str] = False,
        analyze: Optional[str] = False,
    ) -> str:
        path = []
        if analyze:
            flop_count = 0
            intermediate_size = 0
        for positions, computing_format in computing_path:
            lhsmodes, _ = einsum_expression_to_mode(computing_format)
            shapes = [(TensorExpression.__size,) * len(lhsmode) for lhsmode in lhsmodes]
            subpath, subpath_info = oe.contract_path(
                computing_format,
                *shapes,
                optimize=optimize,
                shapes=True,
            )
            path.append((positions, subpath))
            if analyze:
                flop_count += subpath_info.opt_cost
                intermediate_size = max(
                    intermediate_size, subpath_info.largest_intermediate
                )
        if analyze:
            return path, (flop_count, intermediate_size)
        return path

    def _construct_adj_matrix(self):
        num = len(self.indices)
        adj_matrix = np.zeros((num, num), dtype=bool)
        for pair in self._pair_dict.values():
            vector = self._pair_vector(pair)
            adj_matrix = np.logical_or(adj_matrix, np.outer(vector, vector))
        return adj_matrix

    _METHOD_ = {
        "exhaustive": exhaustive_search,
        "greedy": greedy_search,
        "bb": branch_and_bound_search,
    }
