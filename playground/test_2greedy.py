import U_stats.tensor_contraction.path as tpath


def test_2greedy_path():
    mode = ["ab", "bc", "cd"]
    expression = tpath.TensorExpression(mode)
    print(expression)

    path, cost = expression.double_greedy_search()
    print("Path:", path)
    print("Cost:", cost)


if __name__ == "__main__":
    test_2greedy_path()
