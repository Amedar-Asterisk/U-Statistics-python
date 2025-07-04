# U-Statistics Python Package

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A Python package for efficient computation of U-statistics and V-statistics via tensor contraction.

## Features

- Efficient computation of U-statistics and V-statistics
- Support for multiple tensor backends (NumPy and PyTorch)
- Both high-level convenience functions and low-level class interfaces
- Optimized tensor operations using `opt_einsum`

## Installation

```bash
pip install u-stats
```

## Requirements

- Python 3.11+
- NumPy ≥ 1.20.0
- opt_einsum ≥ 3.3.0

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this package in your research, please consider citing it:

```bibtex
@software{u_stats_python,
  author = {Zhang, Ruiqi},
  title = {U-Statistics Python Package},
  url = {https://github.com/zrq/U-Statistics-python},
  version = {0.7.0},
  year = {2024}
}
```
