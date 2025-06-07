import itertools
from typing import Dict

BACKEND = {}

ignore = True
try:
    import numpy as np

    BACKEND["np"] = np.einsum
except ImportError:
    raise ImportError("Numpy is not installed. Please install numpy to use U_stats.")


try:
    import torch

    BACKEND["torch"] = torch.einsum

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_device(
        self, tensor: np.ndarray | torch.Tensor, device: torch.device = _DEVICE
    ) -> torch.Tensor:
        if isinstance(tensor, np.ndarray):
            return torch.tensor(tensor, device=device)
        elif isinstance(tensor, torch.Tensor):
            return tensor.to(device)
        else:
            raise TypeError(
                f"Expected a numpy array or torch tensor, but got {type(tensor)}."
            )

    def _torch_mask_tensor(
        ndim: int, dim: int, index1: int, index2: int, device: torch.device = _DEVICE
    ) -> torch.Tensor:
        shape1 = [1] * ndim
        shape1[index1] = dim
        shape2 = [1] * ndim
        shape2[index2] = dim

        idx1 = torch.arange(dim, device=device).reshape(shape1)
        idx2 = torch.arange(dim, device=device).reshape(shape2)
        mask = idx1 == idx2
        mask = mask.broadcast_to((dim,) * ndim)
        return mask

    def _torch_mask_tensors(
        tensors: Dict[int, torch.Tensor],
        sample_size: int,
        device: torch.device = _DEVICE,
    ) -> Dict[int, torch.Tensor]:
        shapes = {index: tensor.dim() for index, tensor in tensors.items()}
        for index, ndim in shapes.items():
            if ndim > 1:
                mask_total = torch.zeros(
                    (sample_size,) * ndim, dtype=torch.bool, device=device
                )
                for i, j in itertools.combinations(range(ndim), 2):
                    mask = _torch_mask_tensor(ndim, sample_size, i, j, device)
                    mask_total |= mask
                mask_total = ~mask_total
                tensors[index] = tensors[index] * mask_total
        return tensors

except ImportError:
    if ignore:
        pass
    else:
        raise ImportError("Torch is not installed. torch.einsum is unavailable.")
