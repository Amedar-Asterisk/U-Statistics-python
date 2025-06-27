from U_stats import ustat
import numpy as np
import torch
import time
from U_stats._utils._backend import set_backend

set_backend("torch", "cuda")

def test_tensor_performance(methods=None, sizes=None, seed=42):
    """
    Test tensor contraction performance across different methods and sizes
    
    Args:
        methods: List of method configurations to test. Each method is a dict with keys:
                'name': display name for the method
                'einsum': True/False
                'dediag': True/False
                If None, tests all 4 methods.
        sizes: List of tensor sizes to test. If None, uses default sizes.
        seed: Random seed for reproducible results
    """
    
    # Default methods if none specified
    if methods is None:
        methods = [
            {'name': 'Path + No Dediag', 'einsum': False, 'dediag': False},
            {'name': 'Path + Dediag', 'einsum': False, 'dediag': True},
            {'name': 'Einsum + Dediag', 'einsum': True, 'dediag': True},
            {'name': 'Einsum + No Dediag', 'einsum': True, 'dediag': False}
        ]
    
    # Default sizes if none specified
    if sizes is None:
        sizes = [100, 200, 300, 500, 700, 1000]
    
    # Define the motif expression
    mode = [["1", "2"], ["2", "3"], ["3", "4"], ["4", "5"], ["5", "6"], ["6", "7"]]
    
    print("=" * 80)
    print("Tensor Contraction Performance Test")
    print(f"Testing {len(methods)} methods on {len(sizes)} different sizes")
    print("=" * 80)
    print(f"{'Size':<8} {'Method':<25} {'Time(s)':<12} {'Result':<15} {'Match':<8}")
    print("-" * 80)
    
    for size in sizes:
        print(f"\nTesting tensor size: {size}x{size}")
        
        # Generate random tensor with fixed seed for reproducibility
        np.random.seed(seed)
        tensor = np.random.rand(size, size)
        tensors = [tensor for _ in range(6)]
        
        results = {}
        times = {}
        
        for i, method in enumerate(methods):
            method_key = f"method_{i}"

            try:
                start_time = time.time()
                result = ustat(
                    tensors=tensors,
                    expression=mode,
                    average=True,
                    path_method="double-greedy-degree-then-fill",
                    _einsum=method['einsum'],
                    _dediag=method['dediag']
                )
                elapsed_time = time.time() - start_time

                results[method_key] = result
                times[method_key] = elapsed_time

            except (MemoryError, RuntimeError) as e:
                results[method_key] = None
                times[method_key] = "-"
                print(f"  ⚠️  Skipped '{method['name']}' due to memory/runtime error.")

            except Exception as e:
                results[method_key] = None
                times[method_key] = "-"
                print(f"  ❗ Unexpected error in '{method['name']}': {e}")

        # Use first method as baseline for comparison
        baseline_key = "method_0"
        baseline_result = results[baseline_key]

        if isinstance(baseline_result, (torch.Tensor, np.ndarray)):
            if hasattr(baseline_result, 'item'):
                baseline_val = baseline_result.item()
            else:
                baseline_val = float(baseline_result)
        elif baseline_result is not None:
            baseline_val = baseline_result
        else:
            baseline_val = None

        for i, method in enumerate(methods):
            method_key = f"method_{i}"
            result_val = results[method_key]
            time_val = times[method_key]

            if result_val is None:
                result_str = "ERROR"
                match = "-"
            else:
                if isinstance(result_val, (torch.Tensor, np.ndarray)):
                    if hasattr(result_val, 'item'):
                        result_val = result_val.item()
                    else:
                        result_val = float(result_val)

                result_str = f"{result_val:.6e}"
                if i == 0:
                    match = "BASE"
                elif baseline_val is not None:
                    match = "✓" if abs(result_val - baseline_val) < 1e-6 * abs(baseline_val) else "✗"
                else:
                    match = "-"

            print(f"{size:<8} {method['name']:<25} {time_val:<12} {result_str:<15} {match:<8}")

    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("BASE = Baseline method")
    print("✓ = Results match baseline")
    print("✗ = Results differ from baseline")


if __name__ == "__main__":
    # Default full test
    
    methods = [
            {'name': 'Path + No Dediag', 'einsum': False, 'dediag': False},
            {'name': 'Einsum + No Dediag', 'einsum': True, 'dediag': False},
            {'name': 'Einsum + Dediag', 'einsum': True, 'dediag': True},
            {'name': 'Path + Dediag', 'einsum': False, 'dediag': True},
        ]
    sizes = [100, 500, 1000, 2000, 3000, 4000]
    test_tensor_performance(methods=methods, sizes=sizes)
    
    # Uncomment below for specific tests:
    # quick_test()
    # test_einsum_only()
    # test_large_sizes()