# U-Statistics-python

## Efficient U-Statistics Computation

**U-statistics** frequently arise in statistics, probability theory, theoretical computer science, economics, statistical physics, and machine learning. It is well known that computing these objects can be computationally intensive. This package leverages the combinatorial structure of U-statistics to reduce computational complexity.

We build on optimized einsum engines—[`numpy.einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html), [`torch.einsum`](https://pytorch.org/docs/stable/generated/torch.einsum.html), and [`opt_einsum`](https://optimized-einsum.readthedocs.io/en/stable/)—to support flexible and efficient computation on both CPU and GPU. You can easily switch backends using `set_backend`.

## Installation

```bash
pip install u-stat
```

## Example Usage

The main function is [`ustat`](https://github.com/Amedar-Asterisk/U-Statistics-python/blob/main/src/u_stats/__init__.py#L92-L131), which computes U-statistics with a syntax similar to `einsum`.
Use `set_backend("torch")` to enable GPU or parallel CPU computation automatically (via PyTorch), or use `"numpy"` for single-core NumPy-based computation.

Here’s an example of computing a **7th-order HOIF-type U-statistic** with complexity \$O(n^3)\$:

$$
U = \frac{1}{n(n-1)\cdots(n-6)} \sum_{1\leq a \neq b \neq c \neq d \neq e \neq f \neq g \leq n} A_{a,b} B_{b,c} C_{c,d} D_{d,e} E_{e,f} F_{f,g}
$$

```python
from u_stats import ustat, set_backend
import numpy as np

set_backend("torch")  # Or use "numpy"

n = 100
expression = "ab,bc,cd,de,ef,fg->"
A = np.random.rand(n, n)
B = np.random.rand(n, n)
C = np.random.rand(n, n)
D = np.random.rand(n, n)
E = np.random.rand(n, n)
F = np.random.rand(n, n)

tensors = [A, B, C, D, E, F]
result = ustat(tensors=tensors, expression=expression, average=True)
```
