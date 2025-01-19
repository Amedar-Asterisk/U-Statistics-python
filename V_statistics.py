import numpy as np
import warnings
from sum_path import SumPath
from utils import standardize_indexes, numbers_to_letters, strings2format


def indexes2pairs(Vindexes: list[int], order=2) -> list[int]:
    compute_sequence = standardize_indexes(
        [
            [Vindexes[i + j] for j in range(order)]
            for i in range(len(Vindexes) - order + 1)
        ]
    )[0]
    return compute_sequence


def indexes2strlst(Vindexes: list[int] | int, order=2) -> list[str]:
    if isinstance(Vindexes, int):
        Vindexes = list(range(Vindexes))
    compute_sequence = indexes2pairs(Vindexes, order)
    return numbers_to_letters(compute_sequence)


def Tensor_sum(
    tensor_list: list | dict,
    compute_list: list | SumPath,
    summor="numpy",
    init=True,
    compute_formats=False,
):
    if init:
        if isinstance(summor, str):
            if summor == "numpy":
                summor = np.einsum
            elif summor == "torch":
                try:
                    import torch

                    summor = torch.einsum
                except ImportError:
                    warnings.warn("torch is not imported, using numpy.einsum.")
                    summor = np.einsum
            else:
                raise ValueError("summor must be 'numpy' or 'torch'.")
        tensor_list = {k: tensor_list[k] for k in range(len(tensor_list))}
        compute_list = SumPath(compute_list)
        if len(tensor_list) != len(compute_list):
            raise ValueError(
                "The number of tensors and the number of compute formats do not match."
            )
        dedup_compute = compute_list.dedup_pair()
        for indice, compute_format in dedup_compute.items():
            tensor_list[indice] = summor(compute_format, tensor_list[indice])
        if compute_formats:
            compute_formats = []
            compute_formats.append(str(compute_list))
            print(compute_formats)
    if len(tensor_list) == 1 and len(compute_list) == 1:
        compute_format = "->".join([compute_list.pair(0), ""])
        if compute_formats:
            compute_formats.append(compute_format)
            return summor(compute_format, tensor_list[0]), compute_formats
        return summor(compute_format, tensor_list[0])
    else:
        contract_index = compute_list.next_contract()
        contract_indices, save_position, contract_compute = compute_list.contract(
            contract_index
        )
        if compute_formats:
            compute_formats.append(contract_compute)
        tensor_list[save_position] = summor(
            contract_compute, *[tensor_list[i] for i in contract_indices]
        )
        contract_indices.remove(save_position)
        for indice in contract_indices:
            tensor_list.pop(indice)
        return Tensor_sum(
            tensor_list,
            compute_list,
            summor=summor,
            init=False,
            compute_formats=compute_formats,
        )


def V_statistics(
    tensor_list: list[np.ndarray],
    V_sequence: list[int] = None,
    compute_formats=False,
):
    if V_sequence is None:
        V_sequence = range(len(tensor_list) + tensor_list[0].ndim - 1)
    order = tensor_list[0].ndim
    compute_list = indexes2strlst(V_sequence, order)
    return Tensor_sum(tensor_list, compute_list, compute_formats=compute_formats)


def V_stats_torch(
    tensor_list: list[np.ndarray],
    V_sequence: list[int] = None,
):
    try:
        import torch
    except ImportError:
        raise ImportError("torch is not imported.")
    if V_sequence is None:
        V_sequence = range(len(tensor_list) + tensor_list[0].ndim - 1)
    order = tensor_list[0].ndim
    compute_list = indexes2strlst(V_sequence, order)
    compute_format = strings2format(compute_list, "")
    try:
        import opt_einsum
    except ImportError:
        warnings.warn(
            "opt_einsum is not imported. will contract from left to right (default order)."
        )
    return torch.einsum(
        compute_format, *[torch.from_numpy(tensor) for tensor in tensor_list]
    )
