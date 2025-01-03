import numpy as np
import string
from utils import reverse_mapping, support_list, dedup
from typing import List, Tuple

AB_list = list(string.ascii_lowercase)


class index_table:
    def __init__(self, evl_times: bool = False):
        self._p_dict = {}
        if evl_times:
            self._t_dict = {}

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
        if hasattr(self, "_t_dict"):
            self._eval_times()

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
        if hasattr(self, "._t_dict"):
            self._eval_times()

    def times(self, index: str | list = None):
        return (
            len(self._p_dict[index])
            if index is not None
            else [len(self._p_dict[i]) for i in self._p_dict]
        )

    def locations(self, index: str):
        return self._p_dict[index]

    def _eval_times(self):
        self._t_dict = reverse_mapping({i: len(self._p_dict[i]) for i in self._p_dict})


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
        return "->".join([",".join([self._pair_list[i] for i in range(len(self))]), ""])

    @property
    def pair_sequence(self):
        return self._pair_list.copy()

    @property
    def index_table(self):
        return self._index_table

    @property
    def times_list(self):
        return sorted(self._index_table.times())

    @property
    def indexes(self) -> list[str]:
        return list(self._index_table.indexes)

    def neighbours(self, index: str, pair_position: int):
        return self.pair(pair_position).replace(index, "")

    def pair(self, position: int) -> str:
        return self._pair_list[position]

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

        compute_format = "->".join(
            [
                ",".join([self.pair(position) for position in compute_positions]),
                result,
            ]
        )
        save_position = self.replace(compute_positions, result)
        if hasattr(self, "indicator") and len(self) > 0:
            self.eval()
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
            - self.index_table.times(index)
            + 1
        )

    def eval(self, eval_func: callable = eval_repeatability):
        self.indicator = {}
        for index in self.indexes:
            self.indicator[index] = eval_func(self, index)
        return self.indicator
