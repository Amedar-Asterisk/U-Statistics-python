from typing import Hashable, Sequence

__all__ = [
    "Inputs",
    "Outputs",
    "NestedHashableSequence",
    "HashableSequence",
]

NestedHashableSequence = Sequence[Sequence[Hashable]]
HashableSequence = Sequence[Hashable]

Inputs = NestedHashableSequence
Outputs = HashableSequence
