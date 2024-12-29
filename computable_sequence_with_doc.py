import numpy as np
from tensor_sum import tensor_sum


class index_table:
    """用于管理索引和位置对应关系的表结构。

    这个类维护了一个索引到位置的映射关系,支持添加、删除和查询操作。

    Attributes
    ----------
    _v_l : dict
        存储索引到位置列表的映射
    _v_t : dict
        存储索引出现次数的映射
    """

    def __init__(self):
        """初始化一个空的索引表。"""
        self._v_l = {}
        self._v_t = {}

    @property
    def indexes(self):
        """返回所有索引的集合。

        Returns
        -------
        dict_keys
            包含所有索引的键视图
        """
        return self._v_l.keys()

    def append(self, index: int, location: tuple):
        """添加一个索引及其对应的位置。

        Parameters
        ----------
        index : int
            要添加的索引值
        location : tuple
            索引对应的位置坐标
        """
        if index not in self._v_l:
            self._v_l[index] = []
            self._v_l[index].append(location)
            self._v_t[index] = 1
        else:
            if location not in self._v_l[index]:
                self._v_l[index].append(location)
                self._v_t[index] += 1

    def remove(self, index: int, location: tuple):
        """删除指定索引的特定位置。

        Parameters
        ----------
        index : int
            要删除的索引值
        location : tuple
            要删除的位置坐标
        """
        if index in self._v_l:
            if location in self._v_l[index]:
                self._v_l[index].remove(location)
                self._v_t[index] -= 1
                if self._v_t[index] == 0:
                    del self._v_l[index]
                    del self._v_t[index]

    def times(self, index: int | list = None):
        """获取索引出现的次数。

        Parameters
        ----------
        index : int or list, optional
            要查询的索引或索引列表

        Returns
        -------
        int or list
            索引出现的次数
        """
        if index is None:
            return self._v_t.values()
        if isinstance(index, int):
            return self._v_t[index]
        elif isinstance(index, list):
            return [self._v_t[i] for i in index]
        return self._v_t[index]

    def locations(self, index: int):
        """获取指定索引的所有位置。

        Parameters
        ----------
        index : int
            要查询的索引

        Returns
        -------
        list
            包含所有位置坐标的列表
        """
        return self._v_l[index]


class ComputableSequence:
    """可计算序列的封装类。

    用于处理嵌套序列结构，支持序列的索引管理和操作。

    Parameters
    ----------
    sequence : list
        输入序列，可以是单层嵌套或双层嵌套的列表

    Attributes
    ----------
    _pair_sequence : list
        存储处理后的配对序列
    _index_table : index_table
        管理序列索引的表结构
    """

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
        """返回序列长度。"""
        return len(self._pair_sequence)

    @property
    def pair_sequence(self):
        """返回配对序列。"""
        return self._pair_sequence

    @property
    def index_table(self):
        """返回索引表。"""
        return self._index_table

    @property
    def times_list(self):
        """返回排序后的出现次数列表。"""
        return sorted(self._index_table.times())

    @property
    def indexes(self):
        """返回所有索引。"""
        return self._index_table.indexes

    def pair(self, indice: int):
        """获取指定位置的配对。

        Parameters
        ----------
        indice : int
            位置索引

        Returns
        -------
        tuple
            对应位置的配对值
        """
        return self._pair_sequence[indice]

    def remove(self, pair_indice: int):
        """删除指定位置的配对。

        Parameters
        ----------
        pair_indice : int
            要删除的配对索引

        Returns
        -------
        ComputableSequence
            返回自身实例
        """
        for j in range(len(self._pair_sequence[pair_indice])):
            self._index_table.remove(self.index((pair_indice, j)), (pair_indice, j))
        self._pair_sequence.pop(pair_indice)
        return self

    @staticmethod
    def is_double_nested(lst):
        """检查是否为双层嵌套列表。

        Parameters
        ----------
        lst : list
            要检查的列表

        Returns
        -------
        bool
            是否为双层嵌套
        """
        if isinstance(lst, list):
            return all(
                isinstance(i, list) and not any(isinstance(j, list) for j in i)
                for i in lst
            )
        return False

    @staticmethod
    def is_single_nested(lst):
        """检查是否为单层嵌套列表。

        Parameters
        ----------
        lst : list
            要检查的列表

        Returns
        -------
        bool
            是否为单层嵌套
        """
        if isinstance(lst, list):
            return all(not isinstance(i, list) for i in lst)
        return False

    @staticmethod
    def original2pair(original_sequence: list):
        """将原始序列转换为配对序列。

        Parameters
        ----------
        pair_sequence : list
            原始序列

        Returns
        -------
        list
            转换后的配对序列
        """
        return [
            tuple(original_sequence[i], original_sequence[i + 1])
            for i in range(0, len(original_sequence) - 1)
        ]
