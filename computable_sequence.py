import numpy as np
from tensor_sum import tensor_sum


class index_table:
    def __init__(self):
        self._v_l = {}
        self._v_t = {}

    @property
    def indexes(self):
        return self._v_l.keys()

    def append(self, index: int, location: tuple):
        if index not in self._v_l:
            self._v_l[index] = []
            self._v_l[index].append(location)
            self._v_t[index] = 1
        else:
            if location not in self._v_l[index]:
                self._v_l[index].append(location)
                self._v_t[index] += 1

    def remove(self, index: int, location: tuple):
        if index in self._v_l:
            if location in self._v_l[index]:
                self._v_l[index].remove(location)
                self._v_t[index] -= 1
                if self._v_t[index] == 0:
                    del self._v_l[index]
                    del self._v_t[index]

    def times(self, index: int | list = None):
        if index is None:
            return self._v_t.values()
        if isinstance(index, int):
            return self._v_t[index]
        elif isinstance(index, list):
            return [self._v_t[i] for i in index]
        return self._v_t[index]

    def locations(self, index: int):
        return self._v_l[index]


class ComputableSequence:
    def __init__(self, sequence: list):
        if self.is_double_nested(sequence):
            self._pair_sequence = [tuple(pair) for pair in sequence]
        elif self.is_single_nested(sequence):
            self._pair_sequence = self.original2pair(sequence)
        self._index_table = index_table()
        for i in range(len(self._pair_sequence)):
            for j in range(len(self._pair_sequence[i])):
                self._index_table.append(self.index((i, j), (i, j)))

    def __len__(self):
        return len(self._pair_sequence)

    @property
    def pair_sequence(self):
        return self._pair_sequence

    @property
    def index_table(self):
        return self._index_table

    @property
    def times_list(self):
        return sorted(self._index_table.times())

    @property
    def indexes(self):
        return self._index_table.indexes

    def pair_indexes(self, indice: int):
        return self._pair_sequence[indice]

    def remove(self, pair_indice: int):
        for j in range(len(self._pair_sequence[pair_indice])):
            self._index_table.remove(self.index((pair_indice, j)), (pair_indice, j))
        self._pair_sequence.pop(pair_indice)
        return self

    def index(self, indice: tuple):
        return self._pair_sequence[indice]

    def pair(self, indice: int):
        return self._pair_sequence[indice]

    @staticmethod
    def is_double_nested(lst):
        if isinstance(lst, list):
            return all(
                isinstance(i, list) and not any(isinstance(j, list) for j in i)
                for i in lst
            )
        return False

    @staticmethod
    def is_single_nested(lst):
        if isinstance(lst, list):
            return all(not isinstance(i, list) for i in lst)
        return False

    @staticmethod
    def original2pair(pair_sequence):
        return [
            tuple(pair_sequence[i], pair_sequence[i + 1])
            for i in range(0, len(pair_sequence) - 1)
        ]
