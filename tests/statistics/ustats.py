from U_stats import ustat, UStats, set_backend, get_backend
import numpy as np
import opt_einsum as oe
from time import time


def test_ustat():
    set_backend("torch")  # Set the backend to 'torch' for testing
    # Example tensors
    tensors = [np.random.rand(100, 100, 100), np.random.rand(100, 100, 100)]
    tensors = [get_backend().to_tensor(tensor) for tensor in tensors]

    # Example expression
    expression = "abc,abd->"

    # Calculate U-statistic
    start_time = time()
    result = ustat(tensors=tensors, expression=expression)
    print("Calculation time:", time() - start_time)

    # Print the result
    print("U-statistic result:", result)


if __name__ == "__main__":
    test_ustat()
