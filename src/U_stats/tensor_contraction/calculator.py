import numpy as np
from .path import TensorExpression
from typing import List, Dict, Tuple, Union, Callable, Hashable, Set
from ..utils._typing import Expression

try:
    import torch
except ImportError:
    pass


class TensorContractionCalculator:

    def __init__(self, summor: str = "numpy"):
        self.summor = self._initialize_summor(summor)
        self.summor_name = summor

    def _initialize_summor(self, summor: str) -> Callable:
        if summor == "numpy":
            return np.einsum
        elif summor == "torch":
            import torch

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            return torch.einsum
        else:
            raise ValueError(
                f"Invalid summor: {summor}. It should be either 'numpy' or 'torch'."
            )

    def _initalize_expression(
        self, expression: Union[List[List[Hashable]], List[str]]
    ) -> TensorExpression:
        return TensorExpression(expression)

    def _initalize_tensor_dict(
        self,
        tensors: List[np.ndarray] | Dict[int, np.ndarray],
        shape: Tuple[int, ...],
    ) -> Dict[int, np.ndarray]:
        if self.summor_name == "torch":
            import torch  # noqa: F401

            if isinstance(tensors, list):
                tensors = {
                    i: self._to_device(tensor)
                    for i, tensor in enumerate(tensors)
                    if shape[i] > 0
                }
            elif isinstance(tensors, dict):
                tensors = {
                    i: self._to_device(tensor)
                    for i, tensor in tensors.items()
                    if shape[i] > 0
                }
            return tensors
        elif self.summor_name == "numpy":
            if isinstance(tensors, list):
                return {i: tensor for i, tensor in enumerate(tensors) if shape[i] > 0}
            elif isinstance(tensors, dict):
                return {i: tensor for i, tensor in tensors.items() if shape[i] > 0}
            raise ValueError(
                f"Invalid input: tensors must be a list "
                f"or a dictionary but it is {type(tensors)}."
            )

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
            result = self.summor(format, *tensors)
            for i in positions:
                tensor_dict.pop(i)
            tensor_dict[position_number] = result
            position_number += 1
        if self.summor_name == "numpy":
            return np.prod(list(tensor_dict.values()))
        elif self.summor_name == "torch":
            return np.prod([value.cpu().numpy() for value in tensor_dict.values()])

    def calculate(
        self,
        tensors: List[np.ndarray] | Dict[int, np.ndarray],
        expression: Expression | TensorExpression,
        path_method: str = "greedy",
        print_cost: bool = False,
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
        path, cost = expression.path(path_method)
        if print_cost:
            print(f"The max rank of complexity is {cost}.")
        return self._tensor_contract(tensors, path)

    def _to_device(self, tensor: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(tensor, np.ndarray):
            return torch.tensor(tensor, device=self.device)
        elif isinstance(tensor, torch.Tensor):
            return tensor.to(self.device)
        else:
            raise TypeError(
                f"Expected a numpy array or torch tensor, but got {type(tensor)}."
            )
