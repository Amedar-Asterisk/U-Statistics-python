import numpy as np
from typing import List, Union, Hashable, Dict, Set


def _bool_vector(set: Set[int], length: int) -> np.ndarray:
    vector = np.zeros(length, dtype=bool)
    vector[list(set)] = True
    return vector


set1 = {0, 1, 2}
set2 = {2, 1, 4}

vector1 = _bool_vector(set1, 5)
vector2 = _bool_vector(set2, 5)

print(vector1)
print(vector2)

m1 = np.outer(vector1, vector1)
m2 = np.outer(vector2, vector2)

print(m1)
print(m2)
print(m1 + m2)
