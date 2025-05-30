from U_stats.tensor_contraction.path import TensorExpression
import pytest
import numpy as np

# filepath: U_stats/tensor_contraction/test_path.py


@pytest.fixture
def simple_tensor_expr():
    # Creates a simple tensor expression with mode [['i','j'], ['j','k'], ['k','i']]
    mode = [["i", "j"], ["j", "k"], ["k", "i"]]
    return TensorExpression(mode)


@pytest.fixture
def empty_tensor_expr():
    return TensorExpression(None)


def test_tensor_init(simple_tensor_expr: TensorExpression):
    assert sorted(simple_tensor_expr.indices) == ["a", "b", "c"]
    assert simple_tensor_expr.index_number == 3


def test_empty_tensor_init(empty_tensor_expr: TensorExpression):
    assert empty_tensor_expr.indices == []
    assert empty_tensor_expr.index_number == 0


def test_bool_vector():
    # Test static method _bool_vector
    test_set = {0, 2, 3}
    length = 5
    result = TensorExpression._bool_vector(test_set, length)
    expected = np.array([True, False, True, True, False])
    assert np.array_equal(result, expected)


def test_eliminate_index():
    # Test static method _eliminate_index_
    matrix = np.array([[True, True, False], [True, True, True], [False, True, True]])
    result = TensorExpression._eliminate_index_(matrix, 1)  # eliminate middle row/col
    expected = np.array([[True, True], [True, True]])
    assert np.array_equal(result, expected)


def test_positions(simple_tensor_expr: TensorExpression):
    assert simple_tensor_expr.positions("a") == {0, 2}
    assert simple_tensor_expr.positions("b") == {0, 1}
    assert simple_tensor_expr.positions("c") == {1, 2}
    assert simple_tensor_expr.positions("x") == set()  # non-existent index


def test_evaluate_path(simple_tensor_expr: TensorExpression):
    path = ["a", "b", "c"]
    cost = simple_tensor_expr.evaluate(path)
    assert isinstance(cost, int)
    assert cost >= 0


@pytest.mark.parametrize(
    "path,expected_cost",
    [(["a", "b", "c"], 2), (["b", "c", "a"], 2), (["c", "a", "b"], 2)],
)
def test_evaluate_different_paths(simple_tensor_expr, path, expected_cost):
    assert simple_tensor_expr.evaluate(path) == expected_cost


def test_greedy_search(simple_tensor_expr: TensorExpression):
    path, cost = simple_tensor_expr.greedy_search()
    print(path)
    print(cost)
    assert len(path) == simple_tensor_expr.index_number
    assert cost >= 0
    assert set(path) == set(simple_tensor_expr.indices)


def test_branch_and_bound_search(simple_tensor_expr: TensorExpression):
    path, cost = simple_tensor_expr.branch_and_bound_search()
    assert len(path) == simple_tensor_expr.index_number
    assert cost >= 0
    assert set(path) == set(simple_tensor_expr.indices)

    # Branch and bound should find optimal solution
    _, greedy_cost = simple_tensor_expr.greedy_search()
    assert cost <= greedy_cost


def test_computing_representation(simple_tensor_expr: TensorExpression):
    path = ["i", "j", "k"]
    comp_path = simple_tensor_expr.computing_representation_path(path)
    assert len(comp_path) == len(path)
    for item in comp_path:
        assert len(item) == 3  # Each item should have (index, positions, format)
        assert isinstance(item[0], str)  # index
        assert isinstance(item[1], set)  # positions
        assert isinstance(item[2], str)  # format string


def test_copy(simple_tensor_expr: TensorExpression):
    copied = simple_tensor_expr.copy()
    assert copied.indices == simple_tensor_expr.indices
    assert copied.index_number == simple_tensor_expr.index_number
    assert id(copied) != id(simple_tensor_expr)  # Different objects


def test_complex_tensor():
    # Test with a more complex tensor expression
    mode = [["i", "j", "k"], ["j", "k", "l"], ["k", "l", "m"], ["l", "m", "i"]]
    complex_expr = TensorExpression(mode)

    assert complex_expr.index_number == 5  # i,j,k,l,m

    path, cost = complex_expr.greedy_search()
    assert len(path) == 5
    assert cost >= 0

    # Test that all search methods return valid results
    greedy_path, greedy_cost = complex_expr.greedy_search()
    bnb_path, bnb_cost = complex_expr.branch_and_bound_search()

    assert set(greedy_path) == set(bnb_path) == set(complex_expr.indices)
    assert (
        bnb_cost <= greedy_cost
    )  # Branch and bound should be at least as good as greedy
