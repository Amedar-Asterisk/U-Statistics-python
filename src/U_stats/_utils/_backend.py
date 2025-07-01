"""
Backend Management for U-Statistics Computation
===============================================

This module provides a unified interface for tensor operations across different
backends (NumPy, PyTorch). It allows seamless switching between computation
backends while maintaining the same API.

The Backend class encapsulates all tensor operations needed for U-statistics
computation, including:
- Tensor creation and conversion
- Element-wise operations
- Einstein summation (einsum) operations
- Masking operations for diagonal removal
- Device management (CPU/GPU for PyTorch)

Supported Backends:
    - numpy: CPU-based computation using NumPy arrays
    - torch: GPU/CPU computation using PyTorch tensors

Example:
    >>> from u_stats._utils import set_backend, get_backend
    >>> set_backend("torch")  # Switch to PyTorch backend
    >>> backend = get_backend()
    >>> tensor = backend.to_tensor([1, 2, 3])
"""

import itertools
from typing import Dict, Union, Any, Callable, Optional, List, Tuple, TypeVar
import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import opt_einsum as oe

TensorType = TypeVar("TensorType", np.ndarray, "torch.Tensor")
ShapeType = Union[Tuple[int, ...], List[int]]
DType = Union[np.dtype, "torch.dtype", None]


class Backend:
    """
    Unified backend interface for tensor operations.

    This class provides a consistent API for tensor operations across different
    backends (NumPy, PyTorch). It handles backend-specific implementations
    transparently and manages device placement for GPU backends.

    Args:
        backend: Name of the backend to use ("numpy" or "torch")

    Raises:
        ValueError: If backend is not supported
        ImportError: If PyTorch backend is requested but not available

    Attributes:
        backend: Current backend name
        device: Computation device (None for NumPy, torch.device for PyTorch)
        previous_backend: Previously used backend (for context management)
    """

    def __init__(self, backend: str = "numpy") -> None:
        self.backend: str = backend.lower()

        if self.backend not in ["numpy", "torch"]:
            raise ValueError(
                f"Unsupported backend: {self.backend}. "
                "Supported backends: 'numpy', 'torch'"
            )

        if self.backend == "torch" and not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not available. "
                "Please install torch to use the 'torch' backend."
            )

        self.previous_backend: Optional["Backend"] = None
        self._init_device()
        self._ops: Dict[str, Dict[str, Callable]] = {
            "numpy": {
                "to_tensor": np.asarray,
                "zeros": np.zeros,
                "sign": np.sign,
                "einsum": lambda eq, *ops, **kwargs: oe.contract(
                    eq, *ops, backend="numpy", **kwargs
                ),
                "prod": np.prod,
                "arange": np.arange,
                "ndim": lambda x: x.ndim,
                "broadcast_to": np.broadcast_to,
            },
            "torch": {
                "to_tensor": self._torch_to_tensor,
                "zeros": lambda shape, dtype=None: torch.zeros(
                    shape, dtype=dtype, device=self.device
                ),
                "sign": lambda x: torch.sign(self.to_tensor(x)),
                "einsum": lambda eq, *ops, **kwargs: oe.contract(
                    eq, *ops, backend="torch", **kwargs
                ),
                "prod": lambda x: torch.prod(self.to_tensor(x).float()),
                "arange": lambda dim: torch.arange(dim, device=self.device),
                "ndim": lambda x: x.dim(),
                "broadcast_to": lambda x, shape: x.broadcast_to(shape),
                "to_numpy": lambda x: x.cpu().numpy(),
            },
        }

    def _init_device(self) -> None:
        """Initialize the computation device for the current backend."""
        if self.backend == "torch":
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is not available. Please install torch.")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None

    def _torch_to_tensor(self, x: Any) -> "torch.Tensor":
        """Convert input to PyTorch tensor and move to appropriate device."""
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.tensor(x, device=self.device)

    def _get_op(self, name: str) -> Callable:
        """Get backend-specific operation implementation."""
        return self._ops[self.backend][name]

    def to_tensor(self, x: Any) -> TensorType:
        """
        Convert input to a tensor using the current backend.

        Args:
            x: Input data to convert to tensor

        Returns:
            Tensor representation using current backend
        """
        return self._get_op("to_tensor")(x)

    def zeros(self, shape: ShapeType, dtype: DType = None) -> TensorType:
        """
        Create a tensor filled with zeros.

        Args:
            shape: Shape of the tensor to create
            dtype: Data type of the tensor elements

        Returns:
            Zero-filled tensor with specified shape and dtype
        """
        return self._get_op("zeros")(shape, dtype)

    def sign(self, x: TensorType) -> TensorType:
        """
        Compute element-wise sign of tensor.

        Args:
            x: Input tensor

        Returns:
            Tensor with element-wise sign (-1, 0, or 1)
        """
        return self._get_op("sign")(x)

    def einsum(self, equation: str, *operands: TensorType, **kwargs) -> TensorType:
        """
        Perform Einstein summation on tensors.

        Args:
            equation: Einstein summation equation string
            *operands: Input tensors for the operation
            **kwargs: Additional arguments for optimization

        Returns:
            Result of Einstein summation
        """
        return self._get_op("einsum")(equation, *operands, **kwargs)

    def prod(
        self, range_tuple: Union[range, List[int], Tuple[int, ...]]
    ) -> Union[int, float]:
        """
        Compute product of numbers in a range or sequence.

        Args:
            range_tuple: Range object or sequence of numbers

        Returns:
            Product of all numbers in the sequence
        """
        if isinstance(range_tuple, range):
            numbers = list(range_tuple)
        else:
            numbers = range_tuple
        return self._get_op("prod")(numbers)

    def generate_mask_tensor(
        self, ndim: int, dim: int, index1: int, index2: int
    ) -> TensorType:
        """
        Generate a boolean mask tensor for diagonal elements.

        Creates a mask that identifies diagonal elements between two specified
        dimensions in a multi-dimensional tensor.

        Args:
            ndim: Number of dimensions in the target tensor
            dim: Size of each dimension
            index1: First dimension index for diagonal comparison
            index2: Second dimension index for diagonal comparison

        Returns:
            Boolean mask tensor with True for diagonal elements
        """
        shape1: List[int] = [1] * ndim
        shape1[index1] = dim
        shape2: List[int] = [1] * ndim
        shape2[index2] = dim

        idx1: TensorType = self._get_op("arange")(dim).reshape(shape1)
        idx2: TensorType = self._get_op("arange")(dim).reshape(shape2)
        mask: TensorType = idx1 == idx2
        return self._get_op("broadcast_to")(mask, (dim,) * ndim)

    def dediag_tensors(
        self, tensors: List[TensorType], sample_size: int
    ) -> List[TensorType]:
        """
        Remove diagonal elements from tensors for U-statistics computation.

        U-statistics require sampling without replacement, which means diagonal
        elements (where the same observation appears in multiple positions)
        must be excluded from the computation.

        Args:
            tensors: List of input tensors to process
            sample_size: Size of the sample dimension

        Returns:
            List of tensors with diagonal elements set to zero
        """
        masks: Dict[int, TensorType] = {}

        for k in range(len(tensors)):
            ndim: int = self._get_op("ndim")(tensors[k])
            if ndim > 1:
                if ndim not in masks:
                    mask_total: TensorType = self._get_op("zeros")(
                        (sample_size,) * ndim, dtype=bool
                    )
                    for i, j in itertools.combinations(range(ndim), 2):
                        mask: TensorType = self.generate_mask_tensor(
                            ndim, sample_size, i, j
                        )
                        mask_total |= mask
                    masks[ndim] = ~mask_total

                tensors[k] = tensors[k] * masks[ndim]

        return tensors

    def to_numpy(self, x: TensorType) -> np.ndarray:
        """
        Convert a tensor to a NumPy array.

        For PyTorch tensors, this moves the tensor to CPU before conversion.
        For NumPy arrays, this ensures the output is a NumPy array.

        Args:
            x: The tensor to convert

        Returns:
            NumPy array representation of the tensor
        """
        if self.backend == "torch":
            return self._get_op("to_numpy")(x)
        return np.asarray(x)

    def __enter__(self) -> "Backend":
        """Context manager entry. Sets this backend as the global backend."""
        global _BACKEND
        self.previous_backend = _BACKEND
        _BACKEND = self
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_value: Optional[Exception],
        traceback: Optional[Any],
    ) -> None:
        """Context manager exit. Restores the previous global backend."""
        global _BACKEND
        _BACKEND = self.previous_backend
        self.previous_backend = None


# Global backend instance
_BACKEND: Backend = Backend("numpy")


def get_backend() -> "Backend":
    """
    Get the current global backend instance.

    Returns:
        The currently active Backend instance

    Example:
        >>> backend = get_backend()
        >>> print(backend.backend)  # prints current backend name
    """
    return _BACKEND


def set_backend(backend_name: str) -> None:
    """
    Set the global backend for tensor operations.

    This function changes the global backend used by all U-statistics
    computations. The change affects all subsequent operations until
    another backend is set or the backend context is changed.

    Args:
        backend_name: Name of the backend ("numpy" or "torch")

    Raises:
        ValueError: If backend_name is not supported
        ImportError: If torch backend is requested but PyTorch is not available

    Example:
        >>> set_backend("torch")  # Switch to PyTorch backend
        >>> # All subsequent operations will use PyTorch
        >>> set_backend("numpy")  # Switch back to NumPy

    Note:
        After calling this function, you should use get_backend() to access
        the current backend, or reimport modules that use the backend.
        For temporary backend changes, consider using the Backend class
        as a context manager instead.
    """
    global _BACKEND
    _BACKEND = Backend(backend_name)
