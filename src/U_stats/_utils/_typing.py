from typing import Sequence, Union, Hashable, List, Tuple, Dict
from opt_einsum.typing import ContractionListType, ArrayIndexType, PathType

__all__ = [
    "Expression",
    "Path",
    "IndexPath",
    "TupledPath",
    "PathInfo",
    "_HashableExpression",
    "_IntExpression",
    "_StrExpression",
]

_HashableExpression = Sequence[Union[Sequence[Hashable], str]]
_IntExpression = Sequence[Sequence[int]]
_StrExpression = Sequence[Union[str, Sequence[str]]]

Expression = Union[_IntExpression, _StrExpression, _HashableExpression]


Path = List[Tuple[List[int], str]]
IndexPath = List[int]
TupledPath = List[Tuple[List[int], str, List[Tuple[int]]]]


class PathInfo:
    def __init__(
        self,
        contraction_list: ContractionListType = [],
        input_subscripts: str = "",
        output_subscript: str = "",
        indices: ArrayIndexType = [],
        path: PathType = [],
        scale_list: Sequence[int] = [],
        naive_cost: int = 0,
        opt_cost: int = 0,
        size_list: Sequence[int] = [],
        size_dict: Dict[str, int] = {},
    ) -> None:
        self.contraction_list = contraction_list
        self.input_subscripts = input_subscripts
        self.output_subscript = output_subscript
        self.indices = indices
        self.path = path
        self.scale_list = list(scale_list)
        self.naive_cost = naive_cost
        self.opt_cost = opt_cost
        self.size_list = list(size_list)
        self.size_dict = size_dict

    def update(self, path_info: "PathInfo") -> None:
        self.size_list.extend(path_info.size_list)
        self.scale_list.extend(path_info.scale_list)
        self.naive_cost += path_info.naive_cost
        self.opt_cost += path_info.opt_cost

    def __repr__(self) -> str:
        header = ("scaling", "current")

        path_print = [
            f"  Complete contraction:  {self.eq}\n",
            f"         Naive scaling:  {len(self.indices)}\n",
            f"     Optimized scaling:  {max(self.scale_list, default=0)}\n",
            f"      Naive FLOP count:  {self.naive_cost:.3e}\n",
            f"  Optimized FLOP count:  {self.opt_cost:.3e}\n",
            f"   Theoretical speedup:  {self.speedup:.3e}\n",
            f"  Largest intermediate:  {self.largest_intermediate:.3e} elements\n",
            "-" * 80 + "\n",
            "{:>6} {:>22}\n".format(*header),
            "-" * 80,
        ]

        for n, contraction in enumerate(self.contraction_list):
            _, einsum_str, _ = contraction

            size_remaining = max(0, 56 - max(22, len(einsum_str)))

            path_run = (
                self.scale_list[n],
                einsum_str,
                size_remaining,
            )
            path_print.append("\n{:>4} {:>22}".format(*path_run))

        return "".join(path_print)

    @property
    def eq(self) -> str:
        """Return the equation string."""
        return self.input_subscripts

    @property
    def speedup(self) -> float:
        """Return the speedup factor."""
        if self.naive_cost == 0:
            return float("inf")
        return self.opt_cost / self.naive_cost

    @property
    def largest_intermediate(self) -> float:
        """Return the size of the largest intermediate tensor."""
        if not self.size_list:
            return 0.0
        return max(self.size_list)
