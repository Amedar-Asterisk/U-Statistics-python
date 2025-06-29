import itertools
from typing import Dict, Union, Any, Callable, Optional, List, Tuple, TypeVar
import numpy as np
import opt_einsum as oe
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


TensorType = TypeVar("TensorType", np.ndarray, "torch.Tensor")
ShapeType = Union[Tuple[int, ...], List[int]]
DType = Union[np.dtype, torch.dtype, None]


class Backend:
    def __init__(self, backend: str = "numpy", device: str = None) -> None:
        self.backend: str = backend.lower()
        self.device_str = device 

        if self.backend not in ["numpy", "torch"]:
            raise ValueError(
                f"Unsupported backend: {self.backend}.",
                "Supported backends: 'numpy', 'torch'",
            )

        if self.backend == "torch" and not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not available. ",
                "Please install torch to use the 'torch' backend.",
            )

        self.previous_backend: Optional["Backend"] = None
        self._init_device()
        self._ops: Dict[str, Dict[str, Callable]] = {
            "numpy": {
                "to_tensor": np.asarray,
                "zeros": np.zeros,
                "sign": np.sign,
                "einsum": lambda eq, *ops: oe.contract(
                    eq, *ops, backend="numpy", optimize="greedy"
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
                "einsum": lambda eq, *ops: oe.contract(
                    eq, *ops, backend="torch", optimize="greedy"
                ),
                "prod": lambda x: torch.prod(self.to_tensor(x).float()),
                "arange": lambda dim: torch.arange(dim, device=self.device),
                "ndim": lambda x: x.dim(),
                "broadcast_to": lambda x, shape: x.broadcast_to(shape),
            },
        }

    def _init_device(self) -> None:
        if self.backend == "torch":
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is not available. Please install torch.")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None

    def _torch_to_tensor(self, x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.tensor(x, device=self.device)

    def _get_op(self, name: str) -> Callable:
        return self._ops[self.backend][name]

    def to_tensor(self, x: Any) -> TensorType:
        return self._get_op("to_tensor")(x)

    def zeros(self, shape: ShapeType, dtype: DType = None) -> TensorType:
        return self._get_op("zeros")(shape, dtype)

    def sign(self, x: TensorType) -> TensorType:
        return self._get_op("sign")(x)

    def einsum(self, equation: str, *operands: TensorType) -> TensorType:
        return self._get_op("einsum")(equation, *operands)

    def prod(
        self, range_tuple: Union[range, List[int], Tuple[int, ...]]
    ) -> Union[int, float]:
        if isinstance(range_tuple, range):
            numbers = list(range_tuple)
        else:
            numbers = range_tuple
        return self._get_op("prod")(numbers)

    def generate_mask_tensor(
        self, ndim: int, dim: int, index1: int, index2: int
    ) -> TensorType:
        shape1: List[int] = [1] * ndim
        shape1[index1] = dim
        shape2: List[int] = [1] * ndim
        shape2[index2] = dim

        idx1: TensorType = self._get_op("arange")(dim).reshape(shape1)
        idx2: TensorType = self._get_op("arange")(dim).reshape(shape2)
        mask: TensorType = idx1 == idx2
        return self._get_op("broadcast_to")(mask, (dim,) * ndim)

    def dediag_tensors(
        self, tensors: Dict[int, TensorType], sample_size: int
    ) -> Dict[int, TensorType]:
        masks: Dict[int, TensorType] = {}

        for index, tensor in tensors.items():
            ndim: int = self._get_op("ndim")(tensor)
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

                tensors[index] = tensors[index] * masks[ndim]

        return tensors

    def __enter__(self) -> "Backend":
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
        global _BACKEND
        _BACKEND = self.previous_backend
        self.previous_backend = None


_BACKEND: Backend = Backend("numpy")


def get_backend() -> "Backend":
    """Get the current global backend instance."""
    return _BACKEND


def set_backend(backend_name: str, device: str = None) -> None:
    """
    Set the global backend for tensor operations.

    Args:
        backend_name: Name of the backend ("numpy" or "torch")
        device: Device for computation (only relevant for torch backend)

    Raises:
        ValueError: If backend_name is not supported
        ImportError: If torch backend is requested but not available

    Note:
        After calling this function, you should use get_backend() to access
        the current backend, or reimport modules that use the backend.
    """
    global _BACKEND
    _BACKEND = Backend(backend_name, device)
