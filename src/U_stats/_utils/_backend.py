import itertools
from typing import Dict, Union, Any, Callable
import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class _Backend:

    def __init__(self, backend="numpy", device="cpu"):
        self.backend = backend.lower()
        if self.backend not in ["numpy", "torch"]:
            raise ValueError("Backend must be either 'numpy' or 'torch'")
        if self.backend == "torch" and not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install it first.")

        self.device = device.lower()
        if self.backend == "torch":
            if self.device is None:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            if self.device not in ["cpu", "cuda"]:
                raise ValueError("Device must be either 'cpu' or 'cuda'")
            if self.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available on this machine")

        self._ops = {
            "numpy": {
                "to_tensor": np.asarray,
                "zeros": np.zeros,
                "sign": np.sign,
                "einsum": np.einsum,
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
                "einsum": lambda eq, *ops: torch.einsum(
                    eq, *[self.to_tensor(op) for op in ops]
                ),
                "prod": lambda x: torch.prod(self.to_tensor(x).float()),
                "arange": lambda dim: torch.arange(dim, device=self.device),
                "ndim": lambda x: x.dim(),
                "broadcast_to": lambda x, shape: x.broadcast_to(shape),
            },
        }

    def _torch_to_tensor(self, x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        return torch.tensor(x, device=self.device)

    def _get_op(self, name: str) -> Callable:
        return self._ops[self.backend][name]

    def to_tensor(self, x):
        return self._get_op("to_tensor")(x)

    def zeros(self, shape, dtype=None):
        return self._get_op("zeros")(shape, dtype)

    def sign(self, x):
        return self._get_op("sign")(x)

    def einsum(self, equation, *operands):
        return self._get_op("einsum")(equation, *operands)

    def prod(self, range_tuple):
        if isinstance(range_tuple, range):
            numbers = list(range_tuple)
        else:
            numbers = range_tuple
        return self._get_op("prod")(numbers)

    def generate_mask_tensor(self, ndim: int, dim: int, index1: int, index2: int):
        shape1 = [1] * ndim
        shape1[index1] = dim
        shape2 = [1] * ndim
        shape2[index2] = dim

        idx1 = self._get_op("arange")(dim).reshape(shape1)
        idx2 = self._get_op("arange")(dim).reshape(shape2)
        mask = idx1 == idx2
        return self._get_op("broadcast_to")(mask, (dim,) * ndim)

    def dediag_tensors(
        self, tensors: Dict[int, Union[np.ndarray, torch.Tensor]], sample_size: int
    ) -> Dict[int, Union[np.ndarray, torch.Tensor]]:
        masks = {}

        for index, tensor in tensors.items():
            ndim = self._get_op("ndim")(tensor)
            if ndim > 1:
                if ndim not in masks:
                    mask_total = self._get_op("zeros")(
                        (sample_size,) * ndim, dtype=bool
                    )
                    for i, j in itertools.combinations(range(ndim), 2):
                        mask = self.generate_mask_tensor(ndim, sample_size, i, j)
                        mask_total |= mask
                    masks[ndim] = ~mask_total

                tensors[index] = tensors[index] * masks[ndim]

        return tensors

    def __enter__(self):
        self.previous_backend = BACKEND
        global BACKEND
        BACKEND = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global BACKEND
        BACKEND = self.previous_backend
        self.previous_backend = None


BACKEND = _Backend("numpy", "cpu")


def set_backend(backend_name, device="cpu"):
    global BACKEND
    BACKEND = _Backend(backend_name, device)
