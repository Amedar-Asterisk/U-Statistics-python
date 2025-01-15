import numpy as np
from utils import reverse_mapping, support_list, dedup, strings2format, analyze_path
from typing import List, Tuple


class index_table:
    def __init__(self):
        self._p_dict = {}

    @property
    def indexes(self):
        return self._p_dict.keys()

    def append(self, index: str, location: int):
        """add a new index to the table
        Args:
            index (int): the index of original sequence
            location (int): the pair position of the index
        """
        if index not in self._p_dict:
            self._p_dict[index] = []
            self._p_dict[index].append(location)
        else:
            if location not in self._p_dict[index]:
                self._p_dict[index].append(location)

    def remove(self, index: str, location: int):
        """remove an index from the table
        Args:
            index (int): the index of original sequence
            location (int): the pair position of the index
        """
        if index in self._p_dict:
            if location in self._p_dict[index]:
                self._p_dict[index].remove(location)
                if len(self._p_dict[index]) == 0:
                    self._p_dict.pop(index)

    def locations(self, index: str):
        return self._p_dict[index]


class SumPath:
    def __init__(
        self,
        inputs: list[str],
    ):
        """
        A class to store the index pairs of the tensor operands
        Args:
            sequence (list): a list of index pairs of each tensor in order. The index pair is a string.
        """
        self._pair_list = {i: pair for i, pair in enumerate(inputs)}
        self._index_table = index_table()
        for i in range(len(self._pair_list)):
            for index in self._pair_list[i]:
                self._index_table.append(index, i)

    def __len__(self):
        return len(self._pair_list)

    def __str__(self):
        return strings2format(self._pair_list.values())

    @property
    def pair_sequence(self):
        return list(self._pair_list.values()).copy()

    @property
    def index_table(self):
        return self._index_table

    @property
    def indexes(self) -> list[str]:
        return list(self._index_table.indexes)

    def neighbours(self, index: str, pair_position: int):
        return self.pair(pair_position).replace(index, "")

    def pair(self, position: int) -> str:
        return self._pair_list[position]

    def times(self, index: str):
        return len(self.index_table.locations(index))

    @support_list
    def remove(self, pair_position: int):
        for index in self.pair(pair_position):
            self._index_table.remove(index, pair_position)
        self._pair_list.pop(pair_position)

    def replace(self, pair_positions: int | list[int], pair: str):
        if isinstance(pair_positions, int):
            for index in self.pair(pair_positions):
                self._index_table.remove(index, pair_positions)
            for index in pair:
                self._index_table.append(index, pair_positions)
            self._pair_list[pair_positions] = pair
        elif isinstance(pair_positions, list):
            pair_positions = pair_positions.copy()
            frontmost_position = min(pair_positions)
            for position in pair_positions:
                for index in self.pair(position):
                    self._index_table.remove(index, position)

            self._pair_list[frontmost_position] = pair
            pair_positions.remove(frontmost_position)

            for index in pair:
                self._index_table.append(index, frontmost_position)

            for position in pair_positions:
                self._pair_list.pop(position)
        return frontmost_position

    def dedup_pair(self):
        compute_format = {}
        for k in range(len(self)):
            result = dedup(self.pair(k))
            if len(result) < len(self.pair(k)):
                compute_format[k] = "->".join([self._pair_list[k], result])
                self._pair_list[k] = result
        return compute_format

    def contract(self, index: str) -> Tuple[List[int], str]:
        compute_positions = sorted(self.index_table.locations(index).copy())
        result = dedup([self.pair(position) for position in compute_positions]).replace(
            index, ""
        )
        compute_format = strings2format(
            [self.pair(position) for position in compute_positions], result
        )
        save_position = self.replace(compute_positions, result)

        return compute_positions, save_position, compute_format

    def eval_repeatability(self, index: str):
        return (
            sum(
                [
                    len(self.pair(position))
                    for position in self.index_table.locations(index)
                ]
            )
            - len(
                set(
                    "".join(self.pair(position))
                    for position in self.index_table.locations(index)
                )
            )
            - self.times(index)
            + 1
        )

    def eval_result_length(self, index: str = None):
        if index is None:
            index_length = {
                index: self.eval_result_length(index) for index in self.indexes
            }
            self.length_indicator = reverse_mapping(index_length)
        else:
            return (
                len(
                    set(
                        "".join(
                            [
                                self.pair(position)
                                for position in self.index_table.locations(index)
                            ]
                        )
                    )
                )
                - 1
            )

    def next_contract(self):
        self.eval_result_length()
        indexes_min_length = self.length_indicator[min(self.length_indicator)]
        if len(indexes_min_length) == 1:
            return indexes_min_length[0]
        else:
            return min(indexes_min_length, key=lambda x: self.times(x))

    def analyze_complexity(self, tensor_dim: int = 10, optimize=False):
        sum_path = self
        native_flops = []
        optimized_flops = []
        inter_elements = []
        while len(sum_path) > 1:
            contract_index = sum_path.next_contract()
            _, _, contract_compute = sum_path.contract(contract_index)
            native_flop, optimized_flop, inter_element = analyze_path(
                contract_compute,
                tensor_dim=tensor_dim,
                optimize=optimize,
            )
            native_flops.append(native_flop)
            optimized_flops.append(optimized_flop)
            inter_elements.append(inter_element)
        native_flop = sum(native_flops)
        # optimized_flop = sum(optimized_flops)
        # inter_element = max(inter_elements)
        return native_flop
