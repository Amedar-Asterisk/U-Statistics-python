import numpy as np
from .path import TensorExpression
from typing import List, Dict, Tuple, Union, Hashable, Set
from .._utils import Expression, get_backend


class TensorContractionCalculator:
    def __init__(self):
        return

    def _initalize_expression(
        self, expression: Union[List[List[Hashable]], List[str]]
    ) -> TensorExpression:
        return TensorExpression(expression)

    def _initalize_tensor_dict(
        self,
        tensors: List[np.ndarray] | Dict[int, np.ndarray],
        shape: Tuple[int, ...],
    ) -> Dict[int, np.ndarray]:
        backend = get_backend()
        if isinstance(tensors, list):
            tensors = {
                i: backend.to_tensor(tensor)
                for i, tensor in enumerate(tensors)
                if shape[i] > 0
            }
        elif isinstance(tensors, dict):
            tensors = {
                i: backend.to_tensor(tensor)
                for i, tensor in tensors.items()
                if shape[i] > 0
            }
        return tensors

    def _validate_inputs(
        self,
        tensors: Dict[int, np.ndarray],
        shape: Tuple[int, ...],
    ) -> None:
        if len(tensors.keys()) != len(shape):
            raise ValueError(
                "The number of tensors does not match the expression shape."
            )
        for i, tensor in tensors.items():
            if len(tensor.shape) != shape[i]:
                raise ValueError(f"Tensor {i} has an invalid shape.")
        n_samples = tensors[0].shape[0]
        for tensor in tensors.values():
            if tensor.shape[0] != n_samples:
                raise ValueError("The number of samples in tensors do not match.")

    def _build_einsum_string(self, expression: TensorExpression) -> str:
        """
        Build einsum string from tensor expression.

        Args:
            expression: TensorExpression containing the contraction pattern

        Returns:
            str: Einsum string format
        """
        tensor_indices = []
        for pos in sorted(expression._pair_dict.keys()):
            tensor_indices.append(expression._pair_dict[pos])
        all_indices = set()
        for tensor_idx_list in tensor_indices:
            all_indices.update(tensor_idx_list)

        sorted_indices = sorted(all_indices)
        index_to_char = {idx: chr(ord("a") + i) for i, idx in enumerate(sorted_indices)}
        input_parts = []
        for tensor_idx_list in tensor_indices:
            input_part = "".join(index_to_char[idx] for idx in tensor_idx_list)
            input_parts.append(input_part)

        index_count = {}
        for tensor_idx_list in tensor_indices:
            for idx in tensor_idx_list:
                index_count[idx] = index_count.get(idx, 0) + 1
        einsum_string = ",".join(input_parts) + "->"

        return einsum_string

    def _tensor_contract(
        self,
        tensor_dict: Dict[int, np.ndarray],
        computing_path: List[Tuple[Set, str]] = None,
        expression: TensorExpression = None,
        use_einsum: bool = False,
    ) -> float:
        """
        Tensor contraction with two modes: path-based or direct einsum.

        Args:
            tensor_dict: Dictionary mapping tensor indices to tensors
            computing_path: List of contraction steps (used when use_einsum=False)
            expression: TensorExpression containing the contraction pattern (used when use_einsum=True)
            use_einsum: bool, whether to use direct einsum method

        Returns:
            float: Result of the tensor contraction
        """
        if use_einsum:
            # Direct tensor contraction using backend einsum with full expression
            if expression is None:
                raise ValueError(
                    "expression parameter is required when use_einsum=True"
                )

            backend = get_backend()

            einsum_string = self._build_einsum_string(expression)


            sorted_indices = sorted(tensor_dict.keys())
            tensors = [tensor_dict[i] for i in sorted_indices]

            result = backend.einsum(einsum_string, *tensors)


            if hasattr(result, "item") and result.ndim == 0:
                return result.item()
            elif hasattr(result, "shape") and len(result.shape) == 0:
                return float(result)
            else:
                return result
        else:
            # Path-based tensor contraction
            if computing_path is None:
                raise ValueError(
                    "computing_path parameter is required when use_einsum=False"
                )

            position_number = max(tensor_dict.keys()) + 1
            for positions, format in computing_path:
                tensors = [tensor_dict[i] for i in positions]
                result = get_backend().einsum(format, *tensors)
                for i in positions:
                    tensor_dict.pop(i)
                tensor_dict[position_number] = result
                position_number += 1

            return get_backend().prod(list(tensor_dict.values()))

    def calculate(
        self,
        tensors: List[np.ndarray] | Dict[int, np.ndarray],
        expression: Expression | TensorExpression,
        path_method: str = "greedy",
        _validate: bool = True,
        _init_expression=True,
        _init_tensor=True,
        use_einsum: bool = False,
    ) -> float:
        if _init_expression:
            expression = self._initalize_expression(expression)
        if use_einsum:
            return get_backend().einsum(str(expression), *tensors)
        if _init_tensor:
            tensors = self._initalize_tensor_dict(tensors, expression.shape)
        if _validate:
            self._validate_inputs(tensors, expression.shape)
        if use_einsum:
            return self._tensor_contract(
                tensors, expression=expression, use_einsum=True
            )
        else:
            path, _ = expression.path(path_method)
            return self._tensor_contract(
                tensors, computing_path=path, expression=expression, use_einsum=False
            )
