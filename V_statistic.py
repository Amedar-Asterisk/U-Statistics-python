import numpy as np
import string
from sum_path import SumPath
from utils import standardize_indexes, numbers_to_letters, AB_talbe


def Vindexes2compute_group(Vindexes: list[int]) -> list[int]:
    compute_sequence = standardize_indexes(
        [[Vindexes[i], Vindexes[i + 1]] for i in range(0, len(Vindexes) - 1)]
    )[0]
    return compute_sequence


def Tensor_sum(
    tensor_list: list | dict,
    compute_list: list | SumPath,
    init=True,
    compute_formats=False,
):
    if init:
        tensor_list = {k: tensor_list[k] for k in range(len(tensor_list))}
        compute_list = SumPath(compute_list)
        dedup_compute = compute_list.dedup_pair()
        for indice, compute_format in dedup_compute.items():
            tensor_list[indice] = np.einsum(compute_format, tensor_list[indice])
        compute_list.eval()
        if compute_formats:
            compute_formats = []
            compute_formats.append(str(compute_list))
            print(compute_formats)
    if len(tensor_list) == 1 and len(compute_list) == 1:
        compute_format = "->".join([compute_list.pair(0), ""])
        if compute_formats:
            compute_formats.append(compute_format)
            return np.einsum(compute_format, tensor_list[0]), compute_formats
        return np.einsum(compute_format, tensor_list[0])
    else:
        contract_index = max(
            compute_list.indexes, key=lambda x: compute_list.indicator[x]
        )
        contract_indices, save_position, contract_compute = compute_list.contract(
            contract_index
        )
        if compute_formats:
            compute_formats.append(contract_compute)
        tensor_list[save_position] = np.einsum(
            contract_compute, *[tensor_list[i] for i in contract_indices]
        )
        contract_indices.remove(save_position)
        for indice in contract_indices:
            tensor_list.pop(indice)
        return Tensor_sum(
            tensor_list, compute_list, False, compute_formats=compute_formats
        )


def V_statistic(
    tensor_list: list[np.ndarray],
    compute_group: list[int] = None,
    compute_formats=False,
):
    if compute_group is None:
        compute_group = list(range(order=len(tensor_list) + tensor_list[0].ndim - 1))
    compute_sequence = Vindexes2compute_group(compute_group)
    compute_list = numbers_to_letters(compute_sequence)
    return Tensor_sum(tensor_list, compute_list, compute_formats=compute_formats)


# Test
if __name__ == "__main__":
    np.random.seed(123)
    n = 40
    m = 6
    A = np.random.rand(n, n)
    tensor_list = []
    for _ in range(m):
        tensor_list.append(A)

    compute_list = ["ab", "ac", "ad", "bc", "bd", "cd"]
    import time

    def method1(tensor_list, compute_list):
        return Tensor_sum(tensor_list, compute_list)

    def method2(tensor_list):
        return np.einsum("ab,ac,ad,bc,bd,cd->", *tensor_list)

    # 测试次数
    n_tests = 100

    # 测试第一种方法
    start_time = time.time()
    for _ in range(n_tests):
        result1 = method1(tensor_list, compute_list)
    time1 = (time.time() - start_time) / n_tests

    # 测试第二种方法
    start_time = time.time()
    for _ in range(n_tests):
        result2 = method2(tensor_list)
    time2 = (time.time() - start_time) / n_tests

    print(f"方法1 (Tensor_sum) 平均执行时间: {time1:.6f} 秒")
    print(f"方法2 (einsum) 平均执行时间: {time2:.6f} 秒")
    print(f"速度比例 (method1/method2): {time1/time2:.2f}")

    # 验证结果是否相同
    print(f"\n结果是否相同: {np.allclose(result1, result2)}")
