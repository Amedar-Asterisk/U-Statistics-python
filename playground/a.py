from U_stats.tensor_contraction.path import TensorExpression

mode = [["i", "j"], ["j", "k"], ["k", "l"]]

te = TensorExpression(mode)
print(te._pair_dict)
print(te._indices)
print(te.index_number)

path1, cost1 = te.greedy_search()
# print(path1)
# print(cost1)

# path2, cost2 = te.branch_and_bound_search()
# print(path2)
# print(cost2)

# path3, cost3 = te.exhausive_search_space()
# print(path3)
# print(cost3)

path1 = te.computing_representation_path(path1)
print(path1)
print(path1[0])
print(path1[1])
print(path1[2])
