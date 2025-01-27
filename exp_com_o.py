from joblib import Parallel, delayed
from U2V import encode_partition, get_all_partitions
from V_statistics import indexes2strlst
from sum_path import SumPath, complexity_order
import numpy as np
import csv, time

n = 100

m_min = 4
m_max = 10

save_path = "compexity_order.csv"
with open(save_path, "w") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "U-order",
            "partition-order",
            "partition",
            "complexity",
            "theoritical_complexity_order",
        ]
    )


def analyze_partition(m, partition, save_path, n=n):
    path_seq = SumPath(indexes2strlst(encode_partition(partition)))
    flop = path_seq.analyze_complexity(tensor_dim=n)
    complexity_order_m = complexity_order(m)
    if flop >= n**complexity_order_m:
        with open(save_path, "a+") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    m,
                    len(partition),
                    partition,
                    f"{flop:.2e}",
                    (complexity_order_m, complexity_order_m + 1),
                ]
            )
    return flop


flops = []
for m in range(m_min, m_max + 1):
    print(f"order {m} V-stats complexity analysis")
    start_time = time.time()
    analyze_partition_m = lambda partition: analyze_partition(m, partition, save_path)
    flops_m = Parallel(n_jobs=-1)(
        delayed(analyze_partition_m)(partition) for partition in get_all_partitions(m)
    )
    end_time = time.time()
    print(
        f"processing time: {end_time}",
        f"order {m} V-stats complexity analysis finished in {end_time - start_time} seconds",
    )
    flop_max = max(flops_m)
    flops.append(flop_max)

print("process finished")
for m in range(m_min, m_max + 1):
    print(f"order {m} V-stats max complexity: {flops[m-m_min]:.2e}")
