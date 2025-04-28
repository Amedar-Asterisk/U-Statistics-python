# from joblib import Parallel, delayed
# from itertools import islice
# from U_stats.statistics.U2V import encode_partition, get_all_partitions
# from U_stats.graph import *
# from U_stats.utils import SequenceWriter
# import csv, time, os
# import os


# m_min = 4
# m_max = 10
# batch_size = 10000

# dir_path = "data"
# csv_path = os.path.join(dir_path, "max_degree.csv")
# h5_path = os.path.join(dir_path, "graph_contraction.h5")

# # initialize csv file
# with open(csv_path, "w") as f:
#     writer = csv.writer(f)
#     writer.writerow(["order\\num_piles"] + [f"{i}" for i in range(1, m_max + 1)])

# path_writer = SequenceWriter(h5_path)


# def eval_func(partition):
#     indexes = encode_partition(partition)
#     k = len(partition)
#     graph = VGraph(indexes)
#     if ContractPath.hash(graph) in ContractPath.Evaled_Graphs:
#         return
#     path = ContractPath(graph, eval=True)
#     return k, path


# for m in range(m_min, m_max + 1):
#     start_time = time.time()
#     degree_dict = dict()
#     partitions = get_all_partitions(m)
#     while True:
#         batch = list(islice(partitions, batch_size))
#         if not batch:
#             break
#         results = Parallel(n_jobs=-1)(
#             delayed(eval_func)(partition) for partition in batch
#         )
#         for result in results:
#             if result:
#                 k, path = result
#                 group_path = f"{m}/{k}"
#                 path_writer.add_obj(path, group_path)
#                 degree_dict[k] = degree_dict.get(k, []) + [path.max_contraction_degree]
#     end_time = time.time()
#     print(
#         f"order {m} V-graph analysis finished in {end_time - start_time} seconds",
#     )
#     for k, degrees in degree_dict.items():
#         degree_dict[k] = max(degrees)
#     with open(csv_path, "a+") as f:
#         writer = csv.writer(f)
#         writer.writerow([m] + [degree_dict.get(i, 0) for i in range(1, m_max + 1)])
