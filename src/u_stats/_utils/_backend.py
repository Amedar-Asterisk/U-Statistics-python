"""
Backend Management for U-Statistics Computation
===============================================

This module provides a unified interface for tensor operations across different
computational backends (NumPy, PyTorch). It enables seamless switching between
computation engines while maintaining a consistent API for U-statistics computation.

The Backend class abstracts away backend-specific implementations and provides:
- Unified tensor creation and type conversion
- Cross-backend Einstein summation operations
- Diagonal masking for U-statistics sampling without replacement
- Automatic device management (CPU/GPU) for accelerated backends
- Context manager support for temporary backend switching

Classes:
    Backend: Main backend interface for tensor operations

Functions:
    get_backend: Retrieve the current global backend instance
    set_backend: Configure the global computational backend

Supported Backends:
    - **numpy**: CPU-based computation using NumPy arrays
    - **torch**: GPU/CPU computation using PyTorch tensors
                with automatic device placement

The module automatically detects available backends and handles
    import errors gracefully.
For PyTorch backend, CUDA acceleration is used when available,
    falling back to CPU otherwise.

Examples
--------
Basic backend usage:

>>> from u_stats._utils import set_backend, get_backend
>>>
>>> # Switch to PyTorch backend
>>> set_backend("torch")
>>> backend = get_backend()
>>>
>>> # Create tensors using the current backend
>>> tensor = backend.to_tensor([1, 2, 3])
>>> zeros = backend.zeros((3, 3))

Context manager for temporary backend changes:

>>> from u_stats._utils._backend import Backend
>>>
>>> # Temporarily use PyTorch backend
>>> with Backend("torch") as torch_backend:
...     tensor = torch_backend.to_tensor([[1, 2], [3, 4]])
...     result = torch_backend.einsum("ij,ji->", tensor, tensor.T)
>>>
>>> # Automatically reverts to previous backend

Cross-backend compatibility:

>>> import numpy as np
>>>
>>> # Works with both NumPy and PyTorch backends
>>> data = np.random.randn(10, 10)
>>> backend = get_backend()
>>> tensor = backend.to_tensor(data)
>>> numpy_result = backend.to_numpy(tensor)  # Always returns NumPy array

Notes
-----
**Backend Selection**: The default backend is NumPy, which provides CPU-based
computation suitable for most use cases. PyTorch backend enables GPU acceleration
when CUDA is available, which can significantly speed up large-scale U-statistics
computations.

**Device Management**: For PyTorch backend, tensors are automatically placed on
the best available device (CUDA GPU if available, otherwise CPU). Device placement
is handled transparently.

**Memory Efficiency**: The module uses opt_einsum for optimal Einstein summation
contraction paths, minimizing memory usage and computational complexity across
all backends.
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
    Unified backend interface for tensor operations in U-statistics computation.

    This class provides a consistent API for tensor operations across different
    computational backends (NumPy, PyTorch). It handles backend-specific
    implementations transparently and manages device placement for GPU-accelerated
    backends automatically.

    The Backend class serves as the core abstraction layer that enables U-statistics
    computations to work seamlessly across different tensor libraries while
    maintaining optimal performance characteristics for each backend.

    Parameters
    ----------
    backend : str, default="numpy"
        Name of the computational backend to use.

        Supported values:
        - **"numpy"**: CPU-based computation using NumPy arrays
        - **"torch"**: GPU/CPU computation using PyTorch tensors

    Attributes
    ----------
    backend : str
        Current backend name (normalized to lowercase).
    device : torch.device or None
        Computation device for PyTorch backend (None for NumPy backend).
        Automatically set to CUDA device if available, otherwise CPU.
    previous_backend : Backend or None
        Previously used backend instance for context manager support.

    Raises
    ------
    ValueError
        If the specified backend is not supported.
    ImportError
        If PyTorch backend is requested but PyTorch is not available.

    Examples
    --------
    Basic backend initialization:

    >>> from u_stats._utils._backend import Backend
    >>>
    >>> # Create NumPy backend
    >>> numpy_backend = Backend("numpy")
    >>> print(numpy_backend.backend)  # Output: "numpy"
    >>>
    >>> # Create PyTorch backend (if available)
    >>> torch_backend = Backend("torch")
    >>> print(torch_backend.device)  # Output: device('cuda:0') or device('cpu')

    Using as context manager:

    >>> # Temporarily switch to PyTorch backend
    >>> with Backend("torch") as backend:
    ...     tensor = backend.to_tensor([[1, 2], [3, 4]])
    ...     result = backend.einsum("ij,ji->", tensor, tensor.T)
    >>> # Automatically reverts to previous backend

    Backend operations:

    >>> backend = Backend("numpy")
    >>>
    >>> # Tensor creation
    >>> data = backend.to_tensor([1, 2, 3])
    >>> zeros = backend.zeros((3, 3))
    >>>
    >>> # Mathematical operations
    >>> signs = backend.sign(data)
    >>> product = backend.prod(range(1, 4))  # 1*2*3 = 6

    Notes
    -----
    **Device Management**: For PyTorch backend, the class automatically detects
    and uses CUDA devices when available. All tensor operations are performed
    on the selected device transparently.

    **Operation Mapping**: The backend maintains a mapping of operation names to
    backend-specific implementations, enabling unified method calls regardless
    of the underlying tensor library.

    **Context Manager**: When used as a context manager, the backend temporarily
    becomes the global backend, automatically reverting to the previous backend
    when the context exits.
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
        """
        Initialize the computation device for the current backend.

        For PyTorch backend, automatically selects CUDA device if available,
        otherwise falls back to CPU. For NumPy backend, sets device to None.
        """
        if self.backend == "torch":
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is not available. Please install torch.")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None

    def _torch_to_tensor(self, x: Any) -> "torch.Tensor":
        """
        Convert input to PyTorch tensor and move to appropriate device.

        Parameters
        ----------
        x : Any
            Input data to convert to PyTorch tensor.

        Returns
        -------
        torch.Tensor
            PyTorch tensor on the appropriate device.
        """
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.tensor(x, device=self.device)

    def _get_op(self, name: str) -> Callable:
        """
        Get backend-specific operation implementation.

        Parameters
        ----------
        name : str
            Name of the operation to retrieve.

        Returns
        -------
        Callable
            Backend-specific implementation of the operation.
        """
        return self._ops[self.backend][name]

    def to_tensor(self, x: Any) -> TensorType:
        """
        Convert input to a tensor using the current backend.

        Parameters
        ----------
        x : Any
            Input data to convert to tensor. Can be Python lists, NumPy arrays,
            PyTorch tensors, or any array-like object.

        Returns
        -------
        TensorType
            Tensor representation using the current backend:
            - **np.ndarray** for NumPy backend
            - **torch.Tensor** for PyTorch backend (placed on appropriate device)

        Examples
        --------
        >>> backend = Backend("numpy")
        >>> tensor = backend.to_tensor([1, 2, 3])
        >>> print(type(tensor))  # <class 'numpy.ndarray'>
        >>>
        >>> # PyTorch backend automatically handles device placement
        >>> torch_backend = Backend("torch")
        >>> tensor = torch_backend.to_tensor([[1, 2], [3, 4]])
        >>> print(tensor.device)  # cuda:0 or cpu
        """
        return self._get_op("to_tensor")(x)

    def zeros(self, shape: ShapeType, dtype: DType = None) -> TensorType:
        """
        Create a tensor filled with zeros.

        Parameters
        ----------
        shape : tuple of int or list of int
            Shape of the tensor to create.
        dtype : dtype, optional
            Data type of the tensor elements. If None, uses backend default.

        Returns
        -------
        TensorType
            Zero-filled tensor with specified shape and dtype.

        Examples
        --------
        >>> backend = Backend("numpy")
        >>> zeros = backend.zeros((3, 4))
        >>> print(zeros.shape)  # (3, 4)
        >>> print(zeros.sum())  # 0.0
        """
        return self._get_op("zeros")(shape, dtype)

    def sign(self, x: TensorType) -> TensorType:
        """
        Compute element-wise sign of tensor.

        Parameters
        ----------
        x : TensorType
            Input tensor.

        Returns
        -------
        TensorType
            Tensor with element-wise sign values:
            - **-1** for negative elements
            - **0** for zero elements
            - **1** for positive elements

        Examples
        --------
        >>> backend = Backend("numpy")
        >>> data = backend.to_tensor([-2, 0, 3])
        >>> signs = backend.sign(data)
        >>> print(signs)  # [-1  0  1]
        """
        return self._get_op("sign")(x)

    def einsum(self, equation: str, *operands: TensorType, **kwargs) -> TensorType:
        """
        Perform Einstein summation on tensors.

        This method leverages opt_einsum for optimized contraction paths,
        providing efficient computation across different backends.

        Parameters
        ----------
        equation : str
            Einstein summation equation string (e.g., 'ij,jk->ik').
        *operands : TensorType
            Input tensors for the Einstein summation operation.
        **kwargs
            Additional arguments passed to opt_einsum.contract():
            - **optimize** : str, optimization strategy ('greedy', 'optimal', etc.)
            - **memory_limit** : int, memory limit for intermediate arrays
            - **use_blas** : bool, whether to use BLAS for matrix operations

        Returns
        -------
        TensorType
            Result tensor from Einstein summation.

        Examples
        --------
        Matrix multiplication:

        >>> backend = Backend("numpy")
        >>> A = backend.to_tensor([[1, 2], [3, 4]])
        >>> B = backend.to_tensor([[5, 6], [7, 8]])
        >>> result = backend.einsum('ij,jk->ik', A, B)
        >>> print(result)  # Matrix multiplication result

        Trace computation:

        >>> matrix = backend.to_tensor([[1, 2], [3, 4]])
        >>> trace = backend.einsum('ii->', matrix)
        >>> print(trace)  # 5 (1 + 4)

        Complex multi-tensor contractions:

        >>> # Third-order tensor contraction
        >>> T1 = backend.zeros((10, 5, 8))
        >>> T2 = backend.zeros((5, 8, 12))
        >>> T3 = backend.zeros((12, 10))
        >>> result = backend.einsum('ijk,jkl,li->i', T1, T2, T3)
        """
        return self._get_op("einsum")(equation, *operands, **kwargs)

    def prod(
        self, range_tuple: Union[range, List[int], Tuple[int, ...]]
    ) -> Union[int, float]:
        """
        Compute product of numbers in a range or sequence.

        Parameters
        ----------
        range_tuple : range or list of int or tuple of int
            Range object or sequence of numbers to multiply.

        Returns
        -------
        int or float
            Product of all numbers in the sequence.

        Examples
        --------
        >>> backend = Backend("numpy")
        >>>
        >>> # Product of range
        >>> result = backend.prod(range(1, 5))  # 1*2*3*4
        >>> print(result)  # 24
        >>>
        >>> # Product of list
        >>> result = backend.prod([2, 3, 4])
        >>> print(result)  # 24
        >>>
        >>> # Empty sequence
        >>> result = backend.prod([])
        >>> print(result)  # 1 (empty product)

        Notes
        -----
        This method is commonly used in U-statistics for computing normalization
        factors, particularly for calculating the number of k-permutations:
        n(n-1)(n-2)...(n-k+1).
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
        Generate a boolean mask tensor for identifying diagonal elements.

        Creates a mask that identifies diagonal elements between two specified
        dimensions in a multi-dimensional tensor. This is used in U-statistics
        computation to exclude diagonal terms (sampling with replacement).

        Parameters
        ----------
        ndim : int
            Number of dimensions in the target tensor.
        dim : int
            Size of each dimension (assumes all dimensions have same size).
        index1 : int
            First dimension index for diagonal comparison (0-indexed).
        index2 : int
            Second dimension index for diagonal comparison (0-indexed).

        Returns
        -------
        TensorType
            Boolean mask tensor with shape (dim,) * ndim.
            True values indicate diagonal elements where index1 == index2.

        Examples
        --------
        Generate mask for 2D tensor diagonal:

        >>> backend = Backend("numpy")
        >>> mask = backend.generate_mask_tensor(ndim=2, dim=3, index1=0, index2=1)
        >>> print(mask.shape)  # (3, 3)
        >>> print(mask)
        # [[True, False, False],
        #  [False, True, False],
        #  [False, False, True]]

        Generate mask for 3D tensor diagonal:

        >>> mask = backend.generate_mask_tensor(ndim=3, dim=4, index1=0, index2=2)
        >>> print(mask.shape)  # (4, 4, 4)
        >>> # True where first and third indices are equal

        Notes
        -----
        This method is essential for U-statistics computation where sampling
        without replacement is required. The generated masks are used to zero
        out diagonal elements in tensor operations, ensuring that the same
        observation doesn't appear multiple times in a single term.
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
        must be excluded from the computation. This method creates appropriate
        masks and applies them to zero out diagonal terms.

        Parameters
        ----------
        tensors : list of TensorType
            List of input tensors to process. All tensors should have the same
            sample size along their first dimension.
        sample_size : int
            Size of the sample dimension (number of observations).

        Returns
        -------
        list of TensorType
            List of tensors with diagonal elements set to zero.
            Tensors with only one dimension are returned unchanged.

        Examples
        --------
        Remove diagonals from 2D tensors:

        >>> import numpy as np
        >>> backend = Backend("numpy")
        >>>
        >>> # Create test tensors
        >>> T1 = backend.to_tensor(np.ones((3, 3)))
        >>> T2 = backend.to_tensor(np.ones((3, 3)))
        >>>
        >>> # Remove diagonal elements
        >>> result = backend.dediag_tensors([T1, T2], sample_size=3)
        >>> print(result[0])
        # [[0, 1, 1],
        #  [1, 0, 1],
        #  [1, 1, 0]]

        Process mixed-dimension tensors:

        >>> T1 = backend.zeros((4, 4, 4))  # 3D tensor
        >>> T2 = backend.zeros((4, 4))     # 2D tensor
        >>> T3 = backend.zeros((4,))       # 1D tensor (unchanged)
        >>>
        >>> result = backend.dediag_tensors([T1, T2, T3], sample_size=4)
        >>> # T1 and T2 have diagonals removed, T3 unchanged

        Notes
        -----
        **Algorithm**: For each tensor with dimension > 1:

        1. Generate masks for all pairs of dimensions using `generate_mask_tensor`
        2. Combine masks with logical OR to identify all diagonal elements
        3. Invert the mask and multiply with the tensor to zero diagonals

        **Efficiency**: Masks are cached by dimension to avoid recomputation
        when processing multiple tensors with the same shape.

        **U-Statistics Context**: This operation is essential for computing
        unbiased U-statistics, as it ensures that each term in the sum
        involves distinct observations (sampling without replacement).
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
        For NumPy arrays, this ensures the output is a NumPy array format.

        Parameters
        ----------
        x : TensorType
            The tensor to convert (NumPy array or PyTorch tensor).

        Returns
        -------
        np.ndarray
            NumPy array representation of the tensor.

        Examples
        --------
        Convert NumPy tensor:

        >>> backend = Backend("numpy")
        >>> tensor = backend.to_tensor([1, 2, 3])
        >>> numpy_array = backend.to_numpy(tensor)
        >>> print(type(numpy_array))  # <class 'numpy.ndarray'>

        Convert PyTorch tensor (including GPU tensors):

        >>> torch_backend = Backend("torch")
        >>> tensor = torch_backend.to_tensor([[1, 2], [3, 4]])
        >>> numpy_array = torch_backend.to_numpy(tensor)
        >>> print(type(numpy_array))  # <class 'numpy.ndarray'>

        Notes
        -----
        This method provides a unified interface for converting tensors back
        to NumPy format, which is useful for:
        - Returning results in a consistent format
        - Interfacing with NumPy-based libraries
        - Saving results to disk in standard formats
        """
        if self.backend == "torch":
            return self._get_op("to_numpy")(x)
        return np.asarray(x)

    def __enter__(self) -> "Backend":
        """
        Context manager entry. Sets this backend as the global backend.

        Returns
        -------
        Backend
            This backend instance for use within the context.
        """
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
        """
        Context manager exit. Restores the previous global backend.

        Parameters
        ----------
        exc_type : type or None
            Exception type if an exception occurred.
        exc_value : Exception or None
            Exception instance if an exception occurred.
        traceback : Any or None
            Traceback object if an exception occurred.
        """
        global _BACKEND
        _BACKEND = self.previous_backend
        self.previous_backend = None


# Global backend instance
_BACKEND: Backend = Backend("numpy")


def get_backend() -> "Backend":
    """
    Get the current global backend instance.

    Returns the currently active Backend instance that will be used for
    all tensor operations in U-statistics computations.

    Returns
    -------
    Backend
        The currently active Backend instance.

    Examples
    --------
    Check current backend:

    >>> backend = get_backend()
    >>> print(backend.backend)  # prints current backend name
    >>> print(backend.device)   # prints device for PyTorch backend

    Use current backend for operations:

    >>> backend = get_backend()
    >>> tensor = backend.to_tensor([1, 2, 3])
    >>> zeros = backend.zeros((3, 3))

    Notes
    -----
    The global backend can be changed using `set_backend()` or temporarily
    overridden using the Backend class as a context manager.
    """
    return _BACKEND


def set_backend(backend_name: str) -> None:
    """
    Set the global backend for tensor operations.

    This function changes the global backend used by all U-statistics
    computations. The change affects all subsequent operations until
    another backend is set or the backend context is changed.

    Parameters
    ----------
    backend_name : str
        Name of the backend to use.

        Supported values:
        - **"numpy"**: CPU-based computation using NumPy arrays
        - **"torch"**: GPU/CPU computation using PyTorch tensors

    Raises
    ------
    ValueError
        If backend_name is not supported.
    ImportError
        If torch backend is requested but PyTorch is not available.

    Examples
    --------
    Switch to PyTorch backend:

    >>> set_backend("torch")  # Switch to PyTorch backend
    >>> # All subsequent U-statistics operations will use PyTorch
    >>> from u_stats import UStats
    >>> ustat = UStats('ij,ji->')
    >>> # Computation will use PyTorch tensors and GPU if available

    Switch back to NumPy:

    >>> set_backend("numpy")  # Switch back to NumPy
    >>> # All subsequent operations will use NumPy arrays

    Compare backend performance:

    >>> import time
    >>> import numpy as np
    >>>
    >>> # Test with NumPy backend
    >>> set_backend("numpy")
    >>> data = [np.random.randn(1000, 1000) for _ in range(2)]
    >>> ustat = UStats('ij,ji->')
    >>> start = time.time()
    >>> result_numpy = ustat.compute(data)
    >>> time_numpy = time.time() - start
    >>>
    >>> # Test with PyTorch backend
    >>> set_backend("torch")
    >>> start = time.time()
    >>> result_torch = ustat.compute(data)
    >>> time_torch = time.time() - start
    >>>
    >>> print(f"NumPy time: {time_numpy:.3f}s")
    >>> print(f"PyTorch time: {time_torch:.3f}s")

    Notes
    -----
    **Persistence**: After calling this function, the backend change persists
    for the entire session until explicitly changed again.

    **Module Imports**: Existing UStats instances will automatically use the
    new backend for their computations.

    **Context Management**: For temporary backend changes, consider using the
    Backend class as a context manager instead:

    >>> with Backend("torch"):
    ...     # Temporarily use PyTorch backend
    ...     result = ustat.compute(data)
    >>> # Automatically reverts to previous backend
    """
    global _BACKEND
    _BACKEND = Backend(backend_name)
