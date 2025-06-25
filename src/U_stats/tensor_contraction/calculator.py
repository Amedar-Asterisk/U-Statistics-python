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

        if isinstance(tensors, list):
            tensors = {
                i: get_backend().to_tensor(tensor)
                for i, tensor in enumerate(tensors)
                if shape[i] > 0
            }
        elif isinstance(tensors, dict):
            tensors = {
                i: get_backend().to_tensor(tensor)
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

    def _tensor_contract(
        self, tensor_dict: Dict[int, np.ndarray], computing_path: List[Tuple[Set, str]]
    ) -> float:
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
    ) -> float:
        if _init_expression:
            expression = self._initalize_expression(expression)
        if _init_tensor:
            tensors = self._initalize_tensor_dict(tensors, expression.shape)
        if _validate:
            self._validate_inputs(tensors, expression.shape)
        path, _ = expression.path(path_method)
        return self._tensor_contract(tensors, path)
