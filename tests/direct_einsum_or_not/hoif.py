from U_stats import ustat, vstat
import numpy as np
import torch
import time
import traceback  # Added traceback module for complete error information
from U_stats._utils._backend import set_backend
from U_stats.statistics.U_statistics import U_stats_loop



def test_tensor_performance(order=7, methods=None, sizes=None, seed=42):
    """
    Test tensor contraction performance across different methods and sizes
    
    Args:
        order: Order of the tensor contraction (e.g., 7 for 7th order). Default is 7.
               This determines the number of tensor pairs in the mode expression.
        methods: List of method configurations to test. Each method is a dict with keys:
                'name': display name for the method
                'einsum': True/False
                'dediag': True/False
                If None, tests all 5 methods.
        sizes: List of tensor sizes to test. If None, uses default sizes.
        seed: Random seed for reproducible results
    """
    
    # Validate order parameter
    if order < 2:
        raise ValueError("Order must be at least 2")
    
    # Default methods if none specified
    if methods is None:
        methods = [
            {'name': 'For loop', 'einsum': True, 'dediag': False},
            {'name': 'Path + No Dediag', 'einsum': False, 'dediag': False},
            {'name': 'Path + Dediag', 'einsum': False, 'dediag': True},
            {'name': 'Einsum + Dediag', 'einsum': True, 'dediag': True},
            {'name': 'Einsum + No Dediag', 'einsum': True, 'dediag': False},

        ]
    
    # Default sizes if none specified
    if sizes is None:
        sizes = [100, 200, 300, 500, 700, 1000]
    
    # Generate the motif expression based on order
    # For order m: [["1", "2"], ["2", "3"], ..., ["m-1", "m"]]
    mode = [[str(i), str(i+1)] for i in range(1, order)]
    
    # Number of tensors needed is order - 1
    num_tensors = order - 1
    
    print("=" * 80)
    print("Tensor Contraction Performance Test")
    print(f"Order: {order} (using {num_tensors} tensors)")
    print(f"Mode expression: {mode}")
    print(f"Testing {len(methods)} methods on {len(sizes)} different sizes")
    print("=" * 80)
    print(f"{'Size':<8} {'Method':<25} {'Time(s)':<12} {'Result':<15} {'Match':<8}")
    print("-" * 80)
    
    for size in sizes:
        print(f"\nTesting tensor size: {size}x{size} (Order {order})")
        
        # Generate random tensors with fixed seed for reproducibility
        np.random.seed(seed)
        tensor = np.random.rand(size, size)
        tensors = [tensor for _ in range(num_tensors)]
        
        results = {}
        times = {}
        
        for i, method in enumerate(methods):
            method_key = f"method_{i}"

            try:
                if method['name'] == 'For loop':
                    # Special case for For loop method
                    start_time = time.time()
                    result = U_stats_loop(
                        tensors=tensors,
                        expression=mode
                    )
                    elapsed_time = time.time() - start_time
                else:
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
                print(f"  ⚠️  Skipped '{method['name']}' due to memory/runtime error:")
                print(f"      Error: {type(e).__name__}: {str(e)}")
                # Uncomment the line below if full error stack trace is needed
                # print(f"      Full traceback:\n{traceback.format_exc()}")

            except Exception as e:
                results[method_key] = None
                times[method_key] = "-"
                print(f"  ❗ Unexpected error in '{method['name']}':")
                print(f"      Error: {type(e).__name__}: {str(e)}")
                print(f"      Full traceback:")
                # Show full error stack trace
                print(traceback.format_exc())

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
    
def test_different_orders(methods, orders, sizes):
    """Test multiple orders with small sizes"""
    print("Testing different orders:")

    for order in orders:
        print(f"\n{'='*50}")
        print(f"Testing Order {order}")
        print(f"{'='*50}")
        try:
            test_tensor_performance(methods = methods, order=order, sizes=sizes)
        except Exception as e:
            print(f"❌ Failed to test order {order}:")
            print(f"   Error: {type(e).__name__}: {str(e)}")
            print(f"   Full traceback:")
            print(traceback.format_exc())
            print(f"   Continuing with next order...")
        
if __name__ == "__main__":

    methods = [
        # {'name': 'For loop', 'einsum': True, 'dediag': False},
        {'name': 'Path + No Dediag', 'einsum': False, 'dediag': False},
        {'name': 'Path + Dediag', 'einsum': False, 'dediag': True},
        {'name': 'Einsum + Dediag', 'einsum': True, 'dediag': True},
        {'name': 'Einsum + No Dediag', 'einsum': True, 'dediag': False},
    ]
    set_backend("torch", "cpu") 
    sizes = [1000]
    orders = [7]
    test_different_orders(methods = methods, orders=orders, sizes=sizes)