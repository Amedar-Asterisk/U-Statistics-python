from .._utils import Expression


def Euler(m: int, r: int = 2) -> Expression:
    if m < r or r < 1:
        raise ValueError(
            "Invalid input: m must be greater than or equal to r, and r "
            "must be greater than 0."
        )
    return [[i + k for k in range(r)] for i in range(m - r + 1)]


def single(m: int) -> Expression:
    return [[i] for i in range(m)]


def trival(m: int) -> Expression:
    return [list(range(m))]
