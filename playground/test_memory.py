from init import *
from pympler import asizeof

# for n in range(100, 1000, 100):
#     A = np.random.rand(n, n, n)
#     mem_A = asizeof.asizeof(A)
#     print(f"Size of A with shape {A.shape}: {mem_A} bytes ({mem_A/1024/1024:.2f} MB)")
#     mem_per_element = mem_A / (n * n * n)
#     print(
#         f"Memory per element: {mem_per_element} bytes ({mem_per_element/1024/1024:.2f} MB)"
#     )

a = 1024**3  # 1 GB
print(f"{a / 8:e}")  # 1 GB in bytes

final = a / 8 * 32

n = 10000

k = np.log(final) / np.log(n)

print(k)
