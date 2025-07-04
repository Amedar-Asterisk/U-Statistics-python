## 0.7.2 (2025-07-04)

### Fix

- update citation section in README.md to be commented out and adjust version reference format in pyproject.toml

### Refactor

- **readme-changelog**: add description in changelog, update version in readme

## 0.7.1 (2025-07-04)

### Fix

- update version reference format in README.md for consistency

### Enhance

- Updated docstrings in _convert.py to provide clearer descriptions of conversion utilities, including examples and usage notes for functions like numbers_to_letters, standardize_indices, strlist_to_einsum_eq, and einsum_eq_to_strlist.
- Improved parameter descriptions in UStats and VStats classes to clarify input tensor requirements, emphasizing compatibility with both np.ndarray and torch.Tensor.
- Removed redundant comments and ensured consistency in terminology across the documentation.

### feat
- improve conversion utilities with detailed docstrings and examples

### refactor: 
- update UStats class parameter type to include torch.Tensor

## 0.7.0 (2025-07-04)

### Feat

- enhance pyproject.toml and README for U-Statistics package; add metadata, dependencies, and detailed documentation

### Fix

- convert tensors to backend tensor format in UStats class

### Refactor

- rename calculate methods to compute in UStats and VStats classes

## 0.6.1 (2025-07-01)

### Fix

- improve error message for complexity calculation in UStats

## 0.6.0 (2025-07-01)

### Feat

- add ComplexityInfo class and complexity calculation to UStats

### Fix

- update type hint for DType and _torch_to_tensor method
- improve warning message for dediagonalization in UStats class

## 0.5.0 (2025-06-30)

### Fix

- update input_subscripts assignment in analyze_path method
- remove unused imports in U_statistics, V_statistics, calculator, and path modules
- correct device assignment in Backend class

### Refactor

- **Ustats-Vstats-tensor_contraction**: remove all path search alg of ours. fully use opt_einsum to perform tensorcontraction

## 0.4.3 (2025-06-25)

### Fix

- **backend-typing**: fix bug of backend; adapt import of type var for pathinfo
- **calculator.py**: remove unused import of Callable from typing

## 0.4.2 (2025-06-24)

### Fix

- **path.py-_backend.py-__init__.py**: add type hint to _backend.py，fix bug of path method 2-greedy-degree-then-fill

## 0.4.1 (2025-06-24)

### Perf

- **all-about-selection-of-backend**: optimiza the selection of backend; use set_backend to select and with Backend(...) to temporarily change backend

## 0.4.0 (2025-06-24)

### Feat

- **path.py-CI**: add checking script to check code not having print; delete print in path.py

## 0.3.0 (2025-06-22)

### Feat

- **statistics-U2V**: add implementation for 3 classical U-stats

## 0.2.1 (2025-06-07)

### Refactor

- **__init__.py**: add interface for U_stats_loop

## 0.2.0 (2025-06-07)

### Feat

- **all**: add function analyzing expression; add interface in __init__.py; complete meta information in __init__.py

### Fix

- **V_stats**: fix a bug
- fix chinese checke script name
- **no_chinese_check.py**: fix script path
- **path.py**: fix typos and bugs

## 0.1.0 (2025-05-30)

### Feat

- **tensor_contraction/path/TensorExpression.double_greedy_search-statistics/V_statistics**: add path searching method: double greedy, first find index with minmum-degree then one with minmum-fill-in; fix bug of order of V-statistics; add commits submit rules
