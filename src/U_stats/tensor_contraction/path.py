from functools import cached_property
from typing import List, Tuple, Dict, Set, Optional, Callable
import itertools
import numpy as np
import opt_einsum as oe
from copy import deepcopy
from dataclasses import dataclass
from .._utils import (
    Expression,
    Path,
    PathInfo,
    standardize_indices,
    numbers_to_letters,
    einsum_equation_to_expression,
    expression_to_einsum_equation,
)


@dataclass
class _IndexRegistry:
    __slots__ = ["_location_map"]

    def __init__(self) -> None:
        """Initialize an empty index registry."""
        self._location_map: Dict[int, Set[int]] = {}

    def copy(self) -> "_IndexRegistry":

        new_registry = _IndexRegistry()
        new_registry._location_map = deepcopy(self._location_map)
        return new_registry

    @property
    def indices(self) -> Set[int]:

        return set(self._location_map.keys())

    def append(self, index: int, location: int) -> None:

        if index not in self._location_map:
            self._location_map[index] = set()
        self._location_map[index].add(location)

    def remove(self, index: int, location: int) -> None:

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

        return sorted(list(self._location_map.get(index, set())))


class TensorExpression:

    __size: int = 10**4

    def __init__(self, expression: Optional[Expression] = None) -> None:
        if expression is not None:
            expression = standardize_indices(expression=expression)
            expression = [item for item in expression if len(item) > 0]
            self.eq = expression_to_einsum_equation(numbers_to_letters(expression)[0])
            self.shape = tuple(len(pair) for pair in expression)
            pair_dict = dict.fromkeys(range(len(expression)), None)
            for i, pair in enumerate(expression):
                pair_dict[i] = pair
            self._pair_dict = pair_dict
            self._index_table: "_IndexRegistry" = _IndexRegistry()
            for i, pair in self._pair_dict.items():
                for index in pair:
                    self._index_table.append(index, i)
            self._position_list = list(range(len(self.shape)))

        else:
            self._pair_dict = {}
            self._index_table: "_IndexRegistry" = _IndexRegistry()
            self.shape = None
            self._position_list = []

    def copy(self) -> "TensorExpression":
        new_state = TensorExpression(None)
        new_state._pair_dict = deepcopy(self._pair_dict)
        new_state._index_table = self._index_table.copy()
        new_state.shape = self.shape
        new_state._position_list = self._position_list.copy()
        return new_state

    def __str__(self) -> str:
        return self.eq

    def __getitem__(self, index: int) -> List[int]:
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

    def greedy_degree_search(self) -> Tuple[List[int], int]:
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

    def greedy_fill_in_search(self) -> Tuple[List[int], int]:
        cost = 0
        path = []
        indices = self.indices.copy()
        adj_matrix = self._adj_matrix.copy()
        np.fill_diagonal(adj_matrix, 0)
        index_mapping = self._construct_index_mapping(indices)

        while indices:
            min_fill = float("inf")
            best_position = -1
            best_degree = -1

            for i, index in enumerate(indices):
                position = index_mapping(index)
                neighbors = np.where(adj_matrix[position])[0]
                d = len(neighbors)

                if d <= 1:
                    fill_in = 0
                else:
                    subgraph = adj_matrix[np.ix_(neighbors, neighbors)]
                    existing_edges = np.sum(subgraph) // 2
                    total_possible_edges = d * (d - 1) // 2
                    fill_in = total_possible_edges - existing_edges

                if fill_in < min_fill or (fill_in == min_fill and d < best_degree):
                    min_fill = fill_in
                    best_position = i
                    best_degree = d

            selected_index = indices.pop(best_position)
            path.append(selected_index)
            cost = max(cost, best_degree)

            position = index_mapping(selected_index)
            adj_matrix = self._eliminate_index_(adj_matrix, position)
            index_mapping = self._construct_index_mapping(indices)

        return path, cost

    def double_greedy_degree_then_fill_search(self) -> Tuple[List[int], int]:
        """
        Elimination ordering heuristic prioritizing minimal degree.

        Selection logic:
            1. Choose nodes with the minimum degree.
            2. If multiple nodes share the minimum degree,
        select the one minimizing fill-in, i.e., the node whose neighbors
        form the most edges among themselves (to minimize new edges added).
        """

        cost = 0
        path = []
        indices = self.indices.copy()
        adj_matrix = self._adj_matrix.copy()
        index_mapping = self._construct_index_mapping(indices)

        # first pass to remove indices with single positions
        for index in indices:
            if len(self.positions(index)) == 1:
                position = index_mapping(index)
                cost = max(cost, self._degree(adj_matrix, position, False))
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

    def double_greedy_fill_then_degree_search(self) -> Tuple[List[int], int]:
        """
        Elimination ordering heuristic prioritizing minimal fill-in.

        Selection logic:
            1. Choose the node with the smallest fill-in
        (i.e., would add the fewest new edges).
            2. If multiple nodes have the same fill-in,
        choose the one with the smallest degree.

        Returns:
            Tuple[List[int], int]: The elimination order
        and an upper bound on treewidth.
        """
        cost = 0
        path = []
        indices = self.indices.copy()
        adj_matrix = self._adj_matrix.copy()
        np.fill_diagonal(adj_matrix, 0)
        index_mapping = self._construct_index_mapping(indices)

        # Preprocessing: eliminate indices appearing only once
        for index in indices.copy():
            if len(self.positions(index)) == 1:
                position = index_mapping(index)
                cost = max(cost, self._degree(adj_matrix, position, False))
                adj_matrix = self._delete_index_(adj_matrix, position)
                indices.remove(index)
                path.append(index)
                index_mapping = self._construct_index_mapping(indices)

        while indices:
            candidates = []

            for pos, index in enumerate(indices):
                neighbors = np.where(adj_matrix[pos])[0]
                degree = len(neighbors)

                if degree <= 1:
                    fill_in = 0
                else:
                    subgraph = adj_matrix[np.ix_(neighbors, neighbors)]
                    existing_edges = np.sum(subgraph) // 2
                    total_possible = degree * (degree - 1) // 2
                    fill_in = total_possible - existing_edges

                candidates.append((pos, index, fill_in, degree))

            # Primary: min fill-in; tie-breaker: min degree
            chosen = min(candidates, key=lambda x: (x[2], x[3]))
            pos, index, fill_in, degree = chosen

            cost = max(cost, degree)
            path.append(index)
            adj_matrix = self._eliminate_index_(adj_matrix, pos)
            indices.pop(pos)
            index_mapping = self._construct_index_mapping(indices)

        return path, cost

    def double_greedy_fill_minus_degree_search(self) -> Tuple[List[int], int]:
        """
        Heuristic elimination ordering using (fill-in - degree)
        cost and a dynamic degree bound (k).

        Selection logic:
            1. Prefer nodes with negative cost = fill-in -
            degree (i.e., low fill-in and degree), and among them,
            pick the one with the smallest degree.
            2. If no negative-cost nodes exist, consider nodes
            with non-negative cost and degree ≤ k-1,
            choosing the one with highest cost
            (i.e., maximum fill saving) and smallest degree.
            3. If no such candidates are available,
            fall back to any node with minimal cost and largest degree.

        Returns:
            Tuple[List[int], int]: The elimination order
        and an upper bound on treewidth.
        """

        cost = 0
        path = []
        indices = self.indices.copy()
        adj_matrix = self._adj_matrix.copy()
        np.fill_diagonal(adj_matrix, 0)
        index_mapping = self._construct_index_mapping(indices)

        # Preprocessing: eliminate indices with only one position
        for index in indices.copy():
            if len(self.positions(index)) == 1:
                position = index_mapping(index)
                cost = max(cost, self._degree(adj_matrix, position, False))
                adj_matrix = self._delete_index_(adj_matrix, position)
                indices.remove(index)
                path.append(index)
                index_mapping = self._construct_index_mapping(indices)

        # Compute initial edge count and derive k
        e = np.sum(adj_matrix) // 2
        k = self._calculate_k(e)
        max_allowed_degree = k - 1

        while indices:
            # If the remaining graph is complete, add the rest
            if adj_matrix.all():
                cost = max(cost, len(indices) - 1)
                path += indices
                break

            candidates = []
            for pos, index in enumerate(indices):
                neighbors = np.where(adj_matrix[pos])[0]
                degree = len(neighbors)

                existing_edges = 0
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        if adj_matrix[neighbors[i], neighbors[j]]:
                            existing_edges += 1

                potential_edges = degree * (degree - 1) // 2
                reduction_cost = -degree + (potential_edges - existing_edges)
                candidates.append((pos, index, degree, reduction_cost))

            # Phase 1: nodes with negative reduction cost
            neg_cost_nodes = [(p, i, d, c) for p, i, d, c in candidates if c < 0]

            # Phase 2: nodes with acceptable degree and non-negative cost
            nonneg_cost_nodes = [
                (p, i, d, c)
                for p, i, d, c in candidates
                if c >= 0 and d <= max_allowed_degree
            ]

            if neg_cost_nodes:
                # Select node with smallest degree and lowest cost
                pos, index, deg, _ = min(neg_cost_nodes, key=lambda x: (x[2], x[3]))
            elif nonneg_cost_nodes:
                # Select node with highest reduction cost and smallest degree
                pos, index, deg, _ = max(nonneg_cost_nodes, key=lambda x: (-x[3], x[2]))
            else:
                # Fallback: any non-negative cost node, smallest cost first
                fallback = [(p, i, d, c) for p, i, d, c in candidates if c >= 0]
                if fallback:
                    pos, index, deg, _ = min(fallback, key=lambda x: (x[3], -x[2]))
                else:
                    break  # No valid nodes left

            path.append(index)
            cost = max(cost, deg)
            adj_matrix = self._eliminate_index_(adj_matrix, pos)
            indices.pop(pos)
            index_mapping = self._construct_index_mapping(indices)

        return path, cost

    def double_greedy_fill_plus_degree_search(self) -> Tuple[List[int], int]:
        """
        Heuristic elimination ordering based on
        score = degree + fill-in with dynamic threshold (k).

        Selection logic:
            1. Prefer nodes with score <= max_allowed_degree + fill-in (tied to k).
            2. Among these, select nodes with minimal score.
            3. If no candidates fit threshold, fallback to minimal score nodes.
            4. Tie-break by smallest degree or highest fill-in accordingly.

        Returns:
            Tuple[List[int], int]: The elimination order
        and an upper bound on treewidth.
        """
        cost = 0
        path = []
        indices = self.indices.copy()
        adj_matrix = self._adj_matrix.copy()
        np.fill_diagonal(adj_matrix, 0)
        index_mapping = self._construct_index_mapping(indices)

        # Preprocessing: eliminate indices appearing in only one position
        for index in indices.copy():
            if len(self.positions(index)) == 1:
                position = index_mapping(index)
                cost = max(cost, self._degree(adj_matrix, position, False))
                adj_matrix = self._delete_index_(adj_matrix, position)
                indices.remove(index)
                path.append(index)
                index_mapping = self._construct_index_mapping(indices)

        # Compute initial edges and threshold k
        e = np.sum(adj_matrix) // 2
        k = self._calculate_k(e)
        max_allowed_score = k + k  # sum of degree + fill-in roughly bounded by 2k

        while indices:
            if adj_matrix.all():
                cost = max(cost, len(indices) - 1)
                path += indices
                break

            candidates = []
            for pos, index in enumerate(indices):
                neighbors = np.where(adj_matrix[pos])[0]
                degree = len(neighbors)

                if degree <= 1:
                    fill_in = 0
                else:
                    subgraph = adj_matrix[np.ix_(neighbors, neighbors)]
                    existing_edges = np.sum(subgraph) // 2
                    total_possible = degree * (degree - 1) // 2
                    fill_in = total_possible - existing_edges

                score = degree + fill_in
                candidates.append((pos, index, degree, fill_in, score))

            # Candidates filtered by score threshold
            filtered = [c for c in candidates if c[4] <= max_allowed_score]
            if filtered:
                # Pick candidate with minimal score, then minimal degree
                chosen = min(filtered, key=lambda x: (x[4], x[2]))
            else:
                # fallback: minimal score, then minimal degree
                chosen = min(candidates, key=lambda x: (x[4], x[2]))

            pos, index, degree, fill_in, score = chosen
            cost = max(cost, degree)
            path.append(index)
            adj_matrix = self._eliminate_index_(adj_matrix, pos)
            indices.pop(pos)
            index_mapping = self._construct_index_mapping(indices)

        return path, cost

    def computing_path(self, path: List[int]) -> Path:
        computing_path = []
        registry = self._index_table.copy()
        pair_dict = self._pair_dict.copy()
        position_number = len(pair_dict)
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

    def path(self, method: str = "2-greedy") -> Tuple[Path, int]:
        if method not in self._METHOD_:
            raise ValueError(
                f"Invalid method: {method}. "
                "Available methods are: {list(self._METHOD_.keys())}"
            )
        index_path, cost = self._METHOD_[method](self)
        computing_path = self.computing_path(index_path)

        return computing_path, cost

    def analyze_path(
        self,
        path: Path,
        size: int = __size,
        optimize: Optional[str] = False,
    ) -> PathInfo:
        path_info = PathInfo()
        path_info.input_subscripts = self.eq
        path_info.output_subscript = ""
        path_info.indices = numbers_to_letters(self.indices)[0]
        path_info.size_dict = {index: size for index in path_info.indices}
        path_info.shapes = [(size,) * len(pair) for pair in self._pair_dict.values()]
        for positions, computing_format in path:
            lhsexpressions, _ = einsum_equation_to_expression(computing_format)
            shapes = [(size,) * len(lhsexpression) for lhsexpression in lhsexpressions]
            subpath, subpath_info = oe.contract_path(
                computing_format,
                *shapes,
                optimize=optimize,
                shapes=True,
            )
            path_info.contraction_list.append((positions, computing_format, subpath))
            path_info.update(subpath_info)
        return path_info

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
    def _degree(adj_matrix: np.ndarray, index: int, with_diag: bool = True) -> int:
        vector = adj_matrix[index]
        return np.sum(vector) - 1 if with_diag else np.sum(vector)

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
            return expression_to_einsum_equation(pairs[:-1]), None
        else:
            pairs = input_pairs + [result_pair]
            pairs, mapping = numbers_to_letters(pairs)
            return expression_to_einsum_equation(pairs[:-1], pairs[-1]), [
                mapping[i] for i in pairs[-1]
            ]

    def _construct_adj_matrix(self):
        num = len(self.indices)
        adj_matrix = np.zeros((num, num), dtype=bool)
        for pair in self._pair_dict.values():
            vector = self._pair_vector(pair)
            adj_matrix = np.logical_or(adj_matrix, np.outer(vector, vector))
        return adj_matrix

    @staticmethod
    def _calculate_k(m: int) -> int:
        """
        Finds the largest integer k such that f(k) <= m,
        where f(k) = binomial coefficient C(k, 2) = k * (k - 1) // 2.

        Args:
            m (int): The threshold value.

        Returns:
            int: The largest integer k such that C(k, 2) <= m.
        """
        k = 0
        while (k * (k - 1)) // 2 <= m:
            k += 1
        return k - 1

    _METHOD_ = {
        "exhaustive": exhaustive_search,
        "bb": branch_and_bound_search,
        "greedy-degree": greedy_degree_search,
        "greedy-fill-in": greedy_fill_in_search,
        "double-greedy-degree-then-fill": double_greedy_degree_then_fill_search,
        "double-greedy-fill-then-degree": double_greedy_fill_then_degree_search,
        "double-greedy-fill-minus-degree": double_greedy_fill_minus_degree_search,
        "double-greedy-fill-plus-degree": double_greedy_fill_plus_degree_search,
    }
