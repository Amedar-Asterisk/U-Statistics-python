##################################################
# Description:
# The ContractPath class is used to calculate the contraction path of a graph using a greedy algorithm.
# The VGraph class is a subclass of networkx.Graph that is used to represent V-statistics graphs.
# The ContractPath class is used to calculate the contraction path of a graph using a greedy algorithm.

import os
import h5py as h5
from typing import List, Optional, Tuple, Set
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from V_statistics import indexes2strlst


class VGraph(nx.Graph):
    """Graph class inherited from networkx.Graph for V-statistics"""

    def __init__(self, indexes: List = None) -> None:
        """
        Initialize V_Graph

        Args:
            indexes: List of indexes to create the graph
        """
        super().__init__()
        if indexes:
            edges = indexes2strlst(indexes)
            self.add_edges_from([edge for edge in edges if edge[0] != edge[1]])


class ContractPath:
    """Class for handling graph contraction paths"""

    Evaled_Graphs = []

    def __init__(self, graph: nx.Graph, eval=True) -> None:
        """
        Initialize ContractPath

        Args:
            graph: Input networkx graph
        """
        if not isinstance(graph, nx.Graph):
            raise TypeError("Input must be a networkx Graph object")

        self._G = graph
        self._hash = self.hash(self._G)
        ContractPath.Evaled_Graphs.append(self._hash)
        self._G_nodes = list(self._G.nodes())
        self._contract_path: List = []
        self._degree_path: List[int] = []
        self._G_path: List[nx.Graph] = []
        if eval:
            self.greedy_path()
            self.contract()

    @property
    def initial_graph(self) -> nx.Graph:
        """Returns a copy of the initial graph"""
        return self._G.copy()

    @property
    def contract_strategy(self) -> List:
        """Returns the contraction strategy"""
        return self._contract_path.copy()

    @property
    def contracting_process(self) -> List[nx.Graph]:
        """Returns the list of graphs during contraction"""
        return self._G_path.copy()

    @property
    def degree_path(self) -> List[int]:
        """Returns the degree path"""
        return self._degree_path.copy()

    @property
    def max_contraction_degree(self) -> int:
        return max(self._degree_path) if self._degree_path else 0

    @property
    def number_of_nodes(self) -> int:
        """Returns the number of nodes in the initial graph"""
        return len(self._G_nodes)

    def greedy_path(self) -> List:
        """
        Calculate contraction path using greedy algorithm

        Returns:
            List of nodes in contraction order
        """
        adj_matrix = nx.to_numpy_array(self._G)
        num_nodes = len(self._G_nodes)
        contract_path = []
        degree_path = []

        available_nodes = set(range(num_nodes))

        while available_nodes:
            min_node_idx = min(
                available_nodes, key=lambda i: np.count_nonzero(adj_matrix[i])
            )

            degree = np.count_nonzero(adj_matrix[min_node_idx])
            contract_path.append(self._G_nodes[min_node_idx])
            degree_path.append(degree)

            D = np.eye(num_nodes)
            D[min_node_idx, min_node_idx] = 0
            adj_matrix = (
                D
                @ (
                    adj_matrix
                    + np.outer(adj_matrix[min_node_idx], adj_matrix[min_node_idx])
                    - np.diag(adj_matrix[min_node_idx])
                )
                @ D
            )

            available_nodes.remove(min_node_idx)

        self._contract_path = contract_path
        self._degree_path = degree_path
        return contract_path

    def contract(self, contract_path: Optional[List] = None) -> List[nx.Graph]:
        """
        Perform graph contraction

        Args:
            contract_path: Optional contraction path, uses greedy path if None

        Returns:
            List of graphs during contraction process
        """
        if contract_path is None:
            if not self._contract_path:
                self.greedy_path()
            contract_path = self._contract_path

        contract_path = contract_path.copy()
        G = self._G.copy()
        self._G_path = [G.copy()]

        while len(contract_path) > 1:
            contracted_node = contract_path.pop(0)
            neighbors = list(G.neighbors(contracted_node))
            G.add_edges_from(combinations(neighbors, 2))
            G.remove_node(contracted_node)
            self._G_path.append(G.copy())

        return self._G_path

    def save(self, filename: str, group: str = None) -> None:
        """
        Save graph state to HDF5 file

        Args:
            filename (str): Path to save the file
            group (str, optional) or group path: Group name in HDF5 file. If is path, divided by '/' to subgroups.
        Return:
            str: Hash value of the saved graph if successful
        """

        with h5.File(filename, "a") as f:
            try:
                if group is None:
                    group = "/"
                group_hierarchy = group.strip("/").split("/")

                file_handle = f
                for group_name in group_hierarchy:
                    if group_name:
                        if group_name not in file_handle:
                            file_handle = file_handle.create_group(group_name)
                        else:
                            file_handle = file_handle[group_name]

                if self._hash in file_handle:
                    raise ValueError(
                        f"Hash {self._hash} already exists in {filename}'s {group} group"
                    )

                save_handle = file_handle.create_group(self._hash)
                save_handle.create_dataset("initial_graph", data=list(self._G.edges()))
                save_handle.create_dataset("contract_path", data=self._contract_path)
                save_handle.create_dataset("degree_path", data=self._degree_path)
                contraction_graph_path = save_handle.create_group("contracting_process")
                for idx, graph in enumerate(self._G_path):
                    contraction_graph_path.create_dataset(
                        str(idx), data=list(graph.edges())
                    )
                return self._hash
            except Exception as e:
                print(e)
                return False

    @classmethod
    def load(cls, filename: str, group: str = None, hash_value: str = None):
        """
        Load graph from HDF5 file

        Args:
            filename (str): Path to the file
            group (str, optional) or group path: Group name in HDF5 file. If is path, divided by '/' to subgroups.
            hash_value (str, optional): Specific hash to load. If None, loads the first available hash

        Returns:
            Graph: Loaded graph object
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File {filename} not found")

        with h5.File(filename, "r") as f:
            try:
                if group is None:
                    group = "/"
                group_hierarchy = group.strip("/").split("/")

                file_handle = f
                for group_name in group_hierarchy:
                    if group_name:
                        if group_name not in file_handle:
                            raise KeyError(f"Group {group_name} not found")
                        file_handle = file_handle[group_name]

                if hash_value:
                    if hash_value not in file_handle:
                        raise KeyError(f"Hash {hash_value} not found")
                    data_group = file_handle[hash_value]
                else:
                    # Get first available hash
                    if len(file_handle.keys()) == 0:
                        raise ValueError("No data found in file")
                    data_group = file_handle[list(file_handle.keys())[0]]

                # Reconstruct initial graph
                G = nx.Graph()
                G.add_edges_from(data_group["initial_graph"][()])

                # Create new graph instance
                graph_obj = cls(G, eval=False)

                # Restore attributes
                graph_obj._contract_path = list(data_group["contract_path"][()])
                graph_obj._degree_path = list(data_group["degree_path"][()])

                # Restore contraction history
                graph_obj._G_path = []
                for idx in range(len(data_group["contracting_process"])):
                    g = nx.Graph()
                    g.add_edges_from(data_group["contracting_process"][str(idx)][()])
                    graph_obj._G_path.append(g)

                return graph_obj

            except Exception as e:
                print(e)
                return False

    @staticmethod
    def hash(graph: nx.Graph) -> str:
        """
        Generate hash for a graph

        Args:
            graph: Input networkx graph

        Returns:
            str: Hash value
        """
        return nx.weisfeiler_lehman_graph_hash(graph, edge_attr=None)


def regular_graph(n, k):
    assert n * k % 2 == 0 and 0 <= k < n, "Invalid parameters"
    yield nx.random_regular_graph(k, n)
