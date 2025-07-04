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
* opt_einsum >= 3.3.0
* torch (optional, required if using the `torch` backend for GPU or parallel CPU computation)

## Example Usage

The main function is [`ustat`](https://github.com/Amedar-Asterisk/U-Statistics-python/blob/main/src/u_stats/__init__.py#L92-L131), which evaluates U-statistics using an einsum-style interface.

Here’s an example of computing a **7th-order U-statistic** with complexity \(O(n^3)\), assuming that the kernel function values have already been precomputed and assembled into matrices \(A, B, C, D, E, F \in \mathbb{R}^{n \times n}\):

```math
U = \frac{1}{n(n-1)\cdots(n-6)} \sum_{1 \leq a \neq b \neq c \neq d \neq e \neq f \neq g \leq n} A_{a,b} B_{b,c} C_{c,d} D_{d,e} E_{e,f} F_{f,g}
```

```python
from u_stats import ustat, set_backend
import numpy as np

# Choose computation backend
set_backend("torch")  # Automatically uses CUDA-enabled GPU if available, otherwise multi-threaded CPU
# set_backend("numpy")  # Use single-threaded NumPy for baseline computation

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

## Backend Options

* `set_backend("numpy")`: Uses NumPy's `einsum`. Single-threaded and deterministic, suitable for small-scale or testing scenarios.
* `set_backend("torch")`: Uses PyTorch's `einsum`. Automatically utilizes CUDA-enabled GPU if available, otherwise falls back to multi-threaded CPU computation. This backend offers significant speedups for large inputs.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! If you find bugs or have feature requests, feel free to open an issue or submit a pull request.
