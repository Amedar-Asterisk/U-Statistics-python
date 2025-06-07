import U_stats
import pytest
import random
from U_stats._utils import einsum_expression_to_mode


@pytest.fixture
def instance_mode():
    return [
        "abc",
        "bde",
        "cde",
        "cdf",
        "bty",
        "btx",
    ]
