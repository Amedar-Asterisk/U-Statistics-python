from typing import (
    Sequence,
    Union,
    Hashable,
)


_HashableExpression = Sequence[Union[Sequence[Hashable], str]]

_IntExpression = Sequence[Sequence[int]]

_StrExpression = Sequence[Union[str, Sequence[str]]]

Expression = Union[_IntExpression, _StrExpression, _HashableExpression]
