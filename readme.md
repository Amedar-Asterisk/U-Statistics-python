# U-Statistics-python

## Efficient U-Statistics Computation

**U-statistics** are fundamental tools in statistics, probability theory, theoretical computer science, economics, statistical physics, and machine learning. However, computing U-statistics can be computationally demanding, especially for high-order cases. This package leverages their underlying combinatorial structure to significantly reduce computational complexity.

We build on optimized einsum engines—[`numpy.einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html), [`torch.einsum`](https://pytorch.org/docs/stable/generated/torch.einsum.html), and [`opt_einsum`](https://optimized-einsum.readthedocs.io/en/stable/)—to enable efficient computation on both CPU and GPU.

## Installation

```bash
pip install u-stat
```

## Requirements

* Python 3.11+
* NumPy >= 1.20.0
* opt\_einsum >= 3.3.0
* torch (optional, required if using the `torch` backend for GPU or parallel CPU computation)

## Example Usage

The main function is [`ustat`](https://github.com/Amedar-Asterisk/U-Statistics-python/blob/main/src/u_stats/__init__.py#L92-L131), which evaluates U-statistics using an einsum-style interface.

Here’s an example of computing a **7th-order U-statistic** with kernel function

$$
h(x_1,x_2,\dots,x_7) = h_1(x_1, x_2) h_2(x_2, x_3) \dots h_6(x_6, x_7),
$$

assuming that the kernel function values have already been precomputed and assembled into matrices $H_1, H_2, \dots, H_6 \in \mathbb{R}^{n \times n}$, then the U-statistic can be computed as follows with complexity $O(n^3)$:

```math
U = \frac{1}{n(n-1)\cdots(n-6)} \sum_{1 \leq a \ne b \ne c \ne d \ne e \ne f \ne g \leq n} H_1[a,b] H_2[b,c] H_3[c,d] H_4[d,e] H_5[e,f] H_6[f,g]
```

The einsum expression corresponding to this example is:

```python
expression = "ab,bc,cd,de,ef,fg->"
```

This follows standard Einstein summation rules: all indices on the left-hand side are summed out if they do not appear on the right. In the context of U-statistics, we restrict the summation to distinct tuples $(a, b, \dots, g)$.

```python
from u_stats import ustat, set_backend
import numpy as np

# Choose computation backend
set_backend("torch")  # Automatically uses CUDA-enabled GPU if available, else CPU
# set_backend("numpy")  # Use single-threaded NumPy for baseline computation

n = 100
expression = "ab,bc,cd,de,ef,fg->"

# Simulated precomputed kernel matrices
H1 = np.random.rand(n, n)
H2 = np.random.rand(n, n)
H3 = np.random.rand(n, n)
H4 = np.random.rand(n, n)
H5 = np.random.rand(n, n)
H6 = np.random.rand(n, n)

tensors = [H1, H2, H3, H4, H5, H6]
result = ustat(tensors=tensors, expression=expression, average=True)
```

## Backend Options

* `set_backend("numpy")`: Uses NumPy's `einsum`. Single-threaded and deterministic, suitable for small-scale or testing scenarios.
* `set_backend("torch")`: Uses PyTorch's `einsum`. Automatically utilizes CUDA-enabled GPU if available, otherwise falls back to multi-threaded CPU computation. This backend offers significant speedups for large inputs.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! If you find bugs or have feature requests, feel free to open an issue or submit a pull request.
