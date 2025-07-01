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


class ComplexityInfo:
    """
    A class to hold the complexity information of a U statistics expression.
    It contains the number of multiplications and the optimal contraction path.
    """

    def __init__(
        self,
        scaling: int = None,
        flops: float = None,
        largest_intermediate: int = None,
    ) -> None:
        self.scaling = 0 if scaling is None else scaling
        self.flops = 0 if flops is None else flops
        self.largest_intermediate = (
            0 if largest_intermediate is None else largest_intermediate
        )

    def update(self, scaling: int, flops: float, largest_intermediate: int) -> None:
        self.scaling = max(self.scaling, scaling)
        self.flops += flops
        self.largest_intermediate = max(self.largest_intermediate, largest_intermediate)
