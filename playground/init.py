import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import U_stats

if __name__ == "__main__":
    print(os.getcwd())
    print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
