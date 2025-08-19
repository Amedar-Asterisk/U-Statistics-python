# U-Statistics-python `u-stats`
[![PyPI version](https://badge.fury.io/py/u-stats.svg)](https://badge.fury.io/py/u-stats)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Style Status](https://img.shields.io/github/actions/workflow/status/Amedar-Asterisk/U-Statistics-python/style_check.yml?branch=main&label=Style)](https://github.com/Amedar-Asterisk/U-Statistics-python/actions)

**U-statistics** are fundamental tools in statistics, probability theory, theoretical computer science, economics, statistical physics, and machine learning. Named after Wassily Hoeffding, U-statistics provide unbiased estimators for population parameters and form the foundation for many statistical tests and methods. However, computing U-statistics can be computationally demanding, especially for high-order cases where the number of combinations grows exponentially.

This package provides a high-performance, tensor-based implementation for computing U-statistics and V-statistics with significant computational advantages:

- Leverages the underlying structure of kernel functions to significantly reduce computational complexity in many cases
- Utilizes optimized einsum engines—[`numpy.einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) and [`torch.einsum`](https://pytorch.org/docs/stable/generated/torch.einsum.html)—to enable efficient computation on both CPU and GPU

If you find this library useful in your research, please consider citing our paper:
Title: On computing and the complexity of computing higher-order U-statistics, exactly
Authors: Xingyu Chen, Ruiqi Zhang And Lin Liu
url: https://arxiv.org/abs/2508.12627
```bibtex
@misc{CZL2025computingU,
  title        = {On computing and the complexity of computing higher-order $U$-statistics, exactly},
  author       = {Xingyu Chen and Ruiqi Zhang and Lin Liu},
  year         = {2025},
  archivePrefix= {arXiv},
  eprint       = {2508.12627},
  primaryClass = {stat.ML},
  doi          = {10.48550/arXiv.2508.12627},
  url          = {https://arxiv.org/abs/2508.12627},
}
```
## Table of Contents

1. [Installation](#1-installation)
2. [Requirements](#2-requirements)
3. [Example Usage](#3-example-usage)
4. [API Reference](#4-api-reference)
5. [Changelog](#5-changelog)
6. [License](#6-license)
7. [Contributing](#7-contributing) 

## 1. Installation

Install the package from PyPI:

```bash
pip install u-stats
```

For development installation:

```bash
git clone https://github.com/Amedar-Asterisk/U-Statistics-python.git
cd U-Statistics-python
pip install -e .
```

## 2. Requirements

### Required Dependencies
- **Python 3.11+**
- **NumPy >= 1.20.0**
- **opt_einsum >= 3.3.0**

### Optional Dependencies
- **torch >= 1.9.0**: GPU acceleration and parallel CPU computation

## 3. Example Usage

### 3.1 Selection of Backend

The package supports two computation backends for different performance needs:

```python
from u_stats import set_backend, get_backend

# Set backend to NumPy (default)
set_backend("numpy")  # CPU computation, deterministic results

# Set backend to PyTorch (optional)
set_backend("torch")  # GPU acceleration and parallel CPU computation

# Check current backend
current_backend = get_backend()
print(f"Current backend: {current_backend}")
```

### 3.2 Computing U-statistics

[`ustat`](https://github.com/Amedar-Asterisk/U-Statistics-python/blob/main/src/u_stats/__init__.py#L92-L131) is the main function for computing U-statistics.

Here we take a **7th-order U-statistic** with kernel function

$$
h(x_1,x_2,\dots,x_7) = h_1(x_1, x_2) h_2(x_2, x_3) \dots h_6(x_6, x_7)
$$

as an example. For samples $X = (X_1, \dots, X_n)$, the U-statistic takes the form 

$$
U = \frac{1}{n(n-1)\cdots (n-6)} \sum_{(i_1,\dots,i_7) \in P_7}h_1(X_{i_1},X_{i_2})\cdots h_6(X_{i_6},X_{i_7})
$$

where $P_7$ denotes all 7-tuples of distinct indices.

#### 3.2.1 Tensor Assembly

We assume that the kernel function values on samples $X$ have been precomputed and assembled into matrices $H_1, H_2, \dots, H_6 \in \mathbb{R}^{n \times n}$:

$$
H_k[i,j] = h_k(X_i, X_j)
$$

#### 3.2.2 Expression Formats

The expression defines how kernel matrices are connected in the computation. We take this U-statistic as an example to explain how to construct expression. 

To express the structure of the kernel function $h(x_1, x_2, \dots, x_7) = h_1(x_1, x_2) \cdot h_2(x_2, x_3) \cdots h_{6}(x_{6}, x_7)$, we assign a unique index to each distinct variable $x_1, x_2, \dots, x_7$. For each factor $h_k(x_{k}, x_{k+1})$, we collect the indices of the variables it depends on into a pair. The sequence of pairs is then ordered according to the order of the factors in the product. 

We can use the following notation to represent this structure: 

**Einstein Summation Notation:**
```python
expression = "ab,bc,cd,de,ef,fg->"
```

**Nested List Notation:**
```python
expression = [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]]
```

**Format Explanation:**
- In Einstein notation: each letter represents an index, each string like `"ab"` represents a factor of the kernel
- In list notation: each sub-list `[i,j]` represents a factor of the kernel
Both formats are equivalent and specify the same computation pattern in our package

#### 3.2.3 Complete Example

```python
from u_stats import ustat, set_backend
import numpy as np

# Choose computation backend
set_backend("torch")  # Use torch if available
# The default is numpy


# Set number of samples
n = 100

# Create precomputed kernel matrices
# In practice, these would be computed from your actual data
H1 = np.random.rand(n, n)
H2 = np.random.rand(n, n)
H3 = np.random.rand(n, n)
H4 = np.random.rand(n, n)
H5 = np.random.rand(n, n)
H6 = np.random.rand(n, n)

tensors = [H1, H2, H3, H4, H5, H6]
expression = "ab,bc,cd,de,ef,fg->"

# Compute the U-statistic
result = ustat(tensors=tensors, expression=expression, average=True)
print(f"7th-order U-statistic: {result}")

# You can also compute the unaveraged sum
sum_result = ustat(tensors=tensors, expression=expression, average=False)
print(f"Sum (before averaging): {sum_result}")
```



## 4. API Reference

### 4.1 Main Functions

#### `ustat(tensors, expression, average=True, optimize="greedy", **kwargs)`
Compute U-statistics from input tensors.

**Parameters:**
- `tensors` (List[np.ndarray]): List of input tensors (numpy arrays or torch tensors)
- `expression` (str | List | Tuple): Tensor contraction expression
  - String format: Einstein summation notation (e.g., "ij,jk->ik")
  - List format: Nested indices (e.g., [[1,2],[2,3]])
- `average` (bool, default=True): Whether to compute average (True) or sum (False)
- `optimize` (str, default="greedy"): Optimization strategy for tensor contraction
  - "greedy": Fast heuristic optimization
  - "optimal": Exhaustive search for optimal contraction order
  - "dp": Dynamic programming approach
- `**kwargs`: Additional keyword arguments passed to `opt_einsum.contract`

**Returns:** 
- `float`: Computed U-statistic value

**Example:**
```python
result = ustat([H1, H2], "ij,jk->", average=True, optimize="optimal")
```

#### `vstat(tensors, expression, average=True, optimize="greedy", **kwargs)`
Compute V-statistics from input tensors.

**Parameters:** Same as `ustat`

**Returns:** Computed V-statistic value

#### `u_stats_loop(tensors, expression)`
Reference implementation using explicit loops (for validation and small computations).

**Note:** This function is primarily for testing and educational purposes. Use `ustats` for production code.

#### `set_backend(backend_name)`
Set the tensor computation backend.

**Parameters:**
- `backend_name` (str): Backend identifier
  - `"numpy"`: Use NumPy backend
  - `"torch"`: Use PyTorch backend

**Example:**
```python
set_backend("torch")  # Switch to PyTorch backend
```

#### `get_backend()`
Get the current tensor computation backend.

**Returns:** 
- `str`: Current backend name ("numpy" or "torch")

### 4.2 Classes

#### `UStats(expression)`
Class-based interface for U-statistics computation with advanced features.

**Parameters:**
- `expression`: Tensor contraction expression (same format as function interface)

**Methods:**
- `compute(tensors, average=True, **kwargs)`: Compute U-statistic
- `complexity_analysis()`: Analyze computational complexity
- `get_contraction_path()`: Get optimized contraction path

**Example:**
```python
ustats_obj = UStats("ij,jk->")
result = ustats_obj.compute([H1, H2], average=True)
complexity = ustats_obj.complexity_analysis()
```

#### `VStats(expression)`  
Class-based interface for V-statistics computation with advanced features.

**Parameters and Methods:** Same as `UStats`

## 5. Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed version history and changes.

## 6. License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## 7. Contributing

We welcome contributions! Here's how you can help:

### Reporting Issues
- Use the [GitHub issue tracker](https://github.com/Amedar-Asterisk/U-Statistics-python/issues)
- Include minimal reproducible examples
- Specify your environment (Python version, OS, backend)

### Development Setup
```bash
# Clone the repository
git clone https://github.com/Amedar-Asterisk/U-Statistics-python.git
cd U-Statistics-python

# Install in development mode
pip install -e ".[test]"
```

### Pull Requests
- Fork the repository and create a feature branch
- Add tests for new functionality
- Ensure all tests pass and type checking succeeds
- Update documentation as needed
- Follow the existing code style

For questions or discussions, feel free to open an issue or reach out to the maintainers.
