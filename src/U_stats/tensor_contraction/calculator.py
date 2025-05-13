import numpy as np
import warnings
from .path import TensorExpression
from typing import List, Dict, Tuple, Union, Callable, Hashable

try:
    import torch
except ImportError as e:
    warnings.warn(
        "torch is not installed. Some functionalities may not work as expected."
    )
    torch = None

try:
    import opt_einsum as oe
except ImportError as e:
    warnings.warn(
        "opt_einsum is not installed. Some functionalities may not work as expected."
    )
    oe = None

__BACKEND__ = {"numpy": np.einsum, "torch": torch.einsum, "oe": oe.contract}


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
        if summor in __BACKEND__:
            return __BACKEND__[summor]
        else:
            raise ValueError(
                f"Invalid summor: {summor}. Available options are: {list(__BACKEND__.keys())}"
            )

    def _initalize_mode(
        self, mode: Union[List[List[Hashable]], List[str]]
    ) -> TensorExpression:
        return TensorExpression(mode)

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
        self, tensor_dict: Dict[int, np.ndarray], mode: TensorExpression
    ) -> float:
        """Contract tensors according to the computing sequence."""
        tensor_dict = tensor_dict.copy()
        if len(mode) == 0:
            return np.prod(list(tensor_dict.values()))

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
