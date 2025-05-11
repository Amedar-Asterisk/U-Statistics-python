import numpy as np
import warnings
from .state import TensorContractionState
from typing import List, Dict, Tuple, Union, Callable, Hashable


class TensorContractionCalculator:
    """
    A class for contracting multiple tensors according to specified computation rules.
    """

    def __init__(self, summor: str = "numpy"):
        """
        Initialize TensorContractor with specified tensor contraction backend.

        Args:
            summor: str, either "numpy" or "torch"
        """
        self.summor = self._initialize_summor(summor)

    def _initialize_summor(self, summor: str) -> Callable:
        """Set up the tensor contraction function."""
        if isinstance(summor, str):
            if summor == "numpy":
                return np.einsum
            elif summor == "torch":
                try:
                    import torch

                    return torch.einsum
                except ImportError:
                    warnings.warn("torch is not imported, using numpy.einsum.")
                    return np.einsum
            raise ValueError("summor must be 'numpy' or 'torch'.")
        elif callable(summor):
            return summor
        raise ValueError("summor must be a callable function can contract tensors.")

    def _initalize_mode(
        self, mode: Union[List[List[Hashable]], List[str]]
    ) -> TensorContractionState:
        return TensorContractionState(mode)

    def _initalize_tensor_dict(
        self,
        tensors: List[np.ndarray] | Dict[int, np.ndarray],
        shape: Tuple[int, ...],
    ) -> Dict[int, np.ndarray]:
        """Initialize the tensor dictionary."""

        if isinstance(tensors, list):
            return {i: tensor for i, tensor in enumerate(tensors) if shape[i] > 0}
        elif isinstance(tensors, dict):
            return {i: tensor for i, tensor in tensors.items() if shape[i] > 0}
        raise ValueError(
            f"Invalid input: tensors must be a list or a dictionary but it is {type(tensors)}."
        )

    def _validate_inputs(
        self,
        tensors: Dict[int, np.ndarray],
        shape: Tuple[int, ...],
    ) -> None:
        """Validate the input tensors and mode."""
        if len(tensors.keys()) != len(shape):
            raise ValueError("The number of tensors does not match the mode shape.")
        for i, tensor in tensors.items():
            if len(tensor.shape) != shape[i]:
                raise ValueError(f"Tensor {i} has an invalid shape.")
        n_samples = tensors[0].shape[0]
        for tensor in tensors.values():
            if tensor.shape[0] != n_samples:
                raise ValueError("The number of samples in tensors do not match.")

    def _tensor_contract(
        self, tensor_dict: Dict[int, np.ndarray], mode: TensorContractionState
    ) -> float:
        """Contract tensors according to the computing sequence."""
        tensor_dict = tensor_dict.copy()
        if len(mode) == 0:
            return np.prod(list(tensor_dict.values()))

        while len(mode.indexes) > 0:
            contract_index = mode.next_contract()
            contract_indices, save_position, contract_compute = mode.contract(
                contract_index
            )
            tensor_dict[save_position] = self.summor(
                contract_compute, *[tensor_dict[i] for i in contract_indices]
            )

            contract_indices.remove(save_position)
            for indice in contract_indices:
                tensor_dict.pop(indice)

        return np.prod(list(tensor_dict.values()))

    def calculate(
        self,
        tensors: List[np.ndarray] | Dict[int, np.ndarray],
        mode: Union[List[List[Hashable]], List[str], TensorContractionState],
        _validate: bool = True,
        _init_mode=True,
        _init_tensor=True,
    ) -> float:
        """
        Compute tensor contractions iteratively.

        Args:
            tensors: Dictionary mapping indices to tensors
            mode: Sequence of tensor contractions to perform

        Returns:
            float: Result of tensor contractions
        """
        if _init_mode:
            mode = self._initalize_mode(mode)
        if _init_tensor:
            tensors = self._initalize_tensor_dict(tensors, mode.shape)
        if _validate:
            self._validate_inputs(tensors, mode.shape)
        return self._tensor_contract(tensors, mode)
