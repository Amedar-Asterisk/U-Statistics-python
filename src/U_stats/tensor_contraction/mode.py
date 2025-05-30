from typing import List, Tuple, Dict, Set, Union, Hashable
from ..utils import standardize_indexes, numbers_to_letters
from dataclasses import dataclass


@dataclass
class _Mode:
    _data: Tuple[Union[str, Tuple[Hashable, ...]], ...]

    def __post_init__(self):
        if not isinstance(self._data, (list, tuple)):
            raise TypeError("A mode must be a list or tuple")

        if isinstance(self._data, list):
            self._data = tuple(
                item if isinstance(item, str) else tuple(item) for item in self._data
            )

        for item in self._data:
            if isinstance(item, str):
                continue
            if isinstance(item, tuple) and all(
                isinstance(elem, Hashable) for elem in item
            ):
                continue
            raise TypeError(
                "Each item in a mode must be either a string or a tuple of hashable objects"
            )

        self._shape = tuple(len(item) for item in self._data)
        self._order = len(
            {
                elem
                for item in self._data
                for elem in (item if isinstance(item, str) else item)
            }
        )

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def order(self) -> int:
        return self._order

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Union[str, Tuple[Hashable, ...]]:
        return self._data[index]


class _StandardizedMode(_Mode):
    def __init__(self, mode: List[Union[str, List[Hashable]]]):
        standardized_data = standardize_indexes(mode)
        super().__init__(standardized_data)
