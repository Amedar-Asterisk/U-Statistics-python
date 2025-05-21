from init import *

n = 100
A = np.random.rand(n, n)

print(A.shape)
vstater = U_stats.statistics.U_statistics.UStatsCalculator(mode=["ij", "ij"])
vstats = vstater.calculate([A, A])

ustat_loop = U_stats.statistics.U_statistics.U_stats_loop(
    [A, A], [["i", "j"], ["i", "j"]]
)
print((vstats - ustat_loop) / ustat_loop)

# import U_stats.statistics.U2V as U2V

# g = U2V.get_all_partitions({"i", "j"})
# for i in g:
#     print(i)
#     print(U2V.partition_weight(i))
