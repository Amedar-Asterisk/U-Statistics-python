import numpy as np


def generate_symmetric_matrix(n):
    random_matrix = np.random.randint(2, size=(n, n))

    symmetric_matrix = (random_matrix + random_matrix.T) // 2

    return symmetric_matrix


def generate_D(n, j):
    I = np.eye(n)
    e_j = np.zeros(n)
    e_j[j] = 1
    D = I - np.outer(e_j, e_j)
    return D


def generate_E(n, j, k):
    E = np.eye(n)
    E[k, j] = 1
    return E


A = generate_symmetric_matrix(3)
D = generate_D(3, 2)
print(A)
E = generate_E(3, 1, 2)
print(A @ E)
print(E @ A)
print(D @ E.T @ A @ E @ D)
