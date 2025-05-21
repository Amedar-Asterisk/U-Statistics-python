from init import *

import networkx.algorithms.bipartite as nx_bipartite
import networkx.algorithms.euler as nx_euler
from U_stats.utils import *


def line_graph_Knn(n: int) -> nx.Graph:
    return nx.line_graph(nx_bipartite.complete_bipartite_graph(n, n))


if __name__ == "__main__":
    G = line_graph_Knn(3)
    print(len(G.edges()))
    euler_circuit = nx_euler.eulerian_circuit(G)
    standardized_indexes = standardize_indexes(euler_circuit)
    indexes_str = numbers_to_letters(standardized_indexes)
    rep_is = "(" + ", ".join(indexes_str) + ")"
    print(rep_is)
