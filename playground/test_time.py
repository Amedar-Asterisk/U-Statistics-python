from typing import Tuple, List, Union, Any
import numpy as np
import time
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from init import *
from U_stats.statistics.U_statistics import U_stats, UStatsCalculator
from U_stats.statistics.U_statistics import U_stats_loop
import torch

# Timer decorator from the second script
def timer(func):
    def wrapper(*args, **kwargs):
        time1 = time.time()
        result = func(*args, **kwargs)
        time2 = time.time()
        compute_time = time2 - time1
        return result, compute_time
    return wrapper


def produce_data(
    n: int, p: int, loc: float = 0, scale: float = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for U-statistics experiment.

    Args:
        n: Number of samples.
        p: Number of features.
        loc: Mean of the normal distribution. Defaults to 0.
        scale: Standard deviation of the normal distribution. Defaults to 1.

    Returns:
        Tuple containing:
            - X: Generated feature matrix of shape (n, p)
            - A: Binary outcome vector of shape (n,)
    """
    X = np.random.normal(loc=loc, scale=scale, size=(n, p))
    s = round(np.sqrt(p))
    bound = np.sqrt(3 / p)
    alpha = np.random.uniform(-bound, bound, size=p)

    z = X @ alpha
    prob = 1 / (1 + np.exp(-z))
    A = np.random.binomial(1, prob, size=n)
    return X, A


@timer
def get_input_tensor(n: int, kappa: float = 0.5):
    """Create kernel matrix and treatment vector for U-statistics computation.

    Args:
        n: Number of samples.
        kappa: Scale factor for number of features (p = kappa * n).

    Returns:
        Tuple containing:
            - Ker: Kernel matrix of shape (n, n)
            - A: Treatment vector of shape (n,)
    """
    p = int(kappa * n)
    X, A = produce_data(n, p)
    Ker = X @ X.T
    return Ker, A


def get_tensors(m: int, Ker: np.ndarray, A: np.ndarray) -> List[np.ndarray]:
    """Assemble input tensors for U-statistics computation.

    Args:
        m: Number of tensors to generate.
        Ker: Kernel matrix.
        A: Treatment vector.

    Returns:
        List of m tensors where first and last elements are treatment vectors,
        and middle elements are kernel matrices.
    """
    outputs = []
    outputs.append(A)
    for _ in range(m - 1):
        outputs.append(Ker)
    outputs.append(A)
    return outputs


def test_mode(m: int) -> List[Union[int, Tuple[int, int]]]:
    """Generate mode configuration for U-statistics testing.

    Args:
        m: Number of tensors.

    Returns:
        List of modes where first and last elements are single integers,
        and middle elements are pairs of consecutive integers.
    """
    outputs = []
    for i in range(m + 1):
        if i == 0:
            outputs.append([i])
        elif i == m:
            outputs.append([i - 1])
        else:
            outputs.append([i - 1, i])
    return outputs


def prepare_tensors_mode(m: int, Ker: np.ndarray, A: np.ndarray):
    """Prepare input tensors and mode for U-statistics computation.

    args:
        m: Number of tensors.
        Ker: Kernel matrix.
        A: Treatment vector.

    Returns:
        Tuple containing:
            - inputs: List of input tensors for U-statistics computation.
            - mode: Mode configuration for U-statistics testing.
    """
    inputs = get_tensors(m, Ker, A)
    mode = test_mode(m)
    return inputs, mode


@timer
def test_our(tensors, mode, summor="numpy",path_method="greedy_minor"):
    return U_stats(tensors, mode, summor=summor, path_method=path_method)


@timer
def test_loop(tensors, mode):
    return U_stats_loop(tensors, mode)


@timer
def test_nodiag(tensors, mode, summor="numpy", path_method="greedy_minor"):
    tensors = tensors.copy()
    calculator = UStatsCalculator(mode, summor=summor)
    return calculator.caculate_non_diag(tensors, average=True, path_method=path_method)


def run_benchmark(
    m_values=[4, 5, 6, 7, 8],
    n_values=[1000, 2000, 4000],
    repeats=5,
    kappa=0.5,
    summor="torch",
    path_method ="greedy_minor",
    results_dir="benchmark_results",
    run_our=True,  # Flag to control whether to run test_our
    run_nodiag=True  # Flag to control whether to run test_nodiag
):
    """
    Run benchmarks for different combinations of m and n values.
    
    Args:
        m_values: List of m values to test
        n_values: List of n values to test
        repeats: Number of times to repeat each test
        kappa: Scale factor for number of features
        summor: Summation method ("numpy" or "torch")
        results_dir: Directory to save results
        run_our: Whether to run the "our" method
        run_nodiag: Whether to run the "nodiag" method
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize results dictionary
    results = {}
    
    # Try to load existing results if available
    results_file = os.path.join(results_dir, "benchmark_results.json")
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
            print(f"Loaded existing results from {results_file}")
        except Exception as e:
            print(f"Error loading existing results: {e}")
    
    # Run benchmarks for each combination
    for m in m_values:
        for n in n_values:
            # Skip if this combination has already been completed
            key = f"m{m}_n{n}"
            if key in results and len(results[key].get("assemble_times", [])) >= repeats:
                print(f"Skipping m={m}, n={n} (already completed)")
                continue
            
            print(f"Running benchmark for m={m}, n={n}")
            
            # Initialize results for this combination if not already present
            if key not in results:
                results[key] = {
                    "m": m,
                    "n": n,
                    "path_method": path_method,
                    "assemble_times": [],
                    "our_times": [],
                    "nodiag_times": [],
                    "our_results": [],
                    "nodiag_results": []
                }
            
            # Run the benchmark multiple times
            for i in range(len(results[key].get("assemble_times", [])), repeats):
                print(f"  Run {i+1}/{repeats}")
                try:
                    # Get input tensors
                    print("  Starting assembly...")
                    tensors, assemble_time = get_input_tensor(n, kappa)
                    inputs, mode = prepare_tensors_mode(m, *tensors)
                    print(f"  Assembly time: {assemble_time:.4f}s")
                    
                    # Store assembly time
                    results[key]["assemble_times"].append(assemble_time)
                    
                    # Run our method if enabled
                    if run_our:
                        print("  Starting test_our...")
                        try:
                            result_our, time_our = test_our(inputs, mode, summor=summor, path_method=path_method)
                            print(f"  test_our time: {time_our:.4f}s")
                            
                            # Convert result to JSON serializable format if needed
                            if isinstance(result_our, (np.ndarray, torch.Tensor)):
                                result_our_serializable = float(result_our)
                            else:
                                result_our_serializable = result_our
                                
                            results[key]["our_times"].append(time_our)
                            results[key]["our_results"].append(result_our_serializable)
                        except Exception as e:
                            print(f"  Error in test_our: {e}")
                    
                    # Run nodiag method if enabled
                    if run_nodiag:
                        print("  Starting test_nodiag...")
                        try:
                            result_nodiag, time_nodiag = test_nodiag(inputs, mode, summor=summor, path_method=path_method)
                            print(f"  test_nodiag time: {time_nodiag:.4f}s")
                            
                            # Convert result to JSON serializable format if needed
                            if isinstance(result_nodiag, (np.ndarray, torch.Tensor)):
                                result_nodiag_serializable = float(result_nodiag)
                            else:
                                result_nodiag_serializable = result_nodiag
                                
                            results[key]["nodiag_times"].append(time_nodiag)
                            results[key]["nodiag_results"].append(result_nodiag_serializable)
                        except Exception as e:
                            print(f"  Error in test_nodiag: {e}")
                    
                    # Save results after each run
                    with open(results_file, "w") as f:
                        json.dump(results, f, indent=2)
                    
                    print(f"  Run completed and saved")
                    
                except Exception as e:
                    print(f"  Error during run: {e}")
                    # Still save the results we have so far
                    with open(results_file, "w") as f:
                        json.dump(results, f, indent=2)
    
    return results


def analyze_results(results, output_dir="benchmark_results"):
    """
    Analyze benchmark results and create tables and plots.
    
    Args:
        results: Dictionary containing benchmark results
        output_dir: Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract unique m and n values
    m_values = sorted(list(set([results[key]["m"] for key in results])))
    n_values = sorted(list(set([results[key]["n"] for key in results])))
    
    # Create dataframes for different times
    assemble_df = pd.DataFrame(index=m_values, columns=n_values)
    our_df = pd.DataFrame(index=m_values, columns=n_values)
    nodiag_df = pd.DataFrame(index=m_values, columns=n_values)
    total_our_df = pd.DataFrame(index=m_values, columns=n_values)
    total_nodiag_df = pd.DataFrame(index=m_values, columns=n_values)
    
    # Create dataframes for results
    our_results_df = pd.DataFrame(index=m_values, columns=n_values)
    nodiag_results_df = pd.DataFrame(index=m_values, columns=n_values)
    
    # Fill dataframes with average times and results
    for key in results:
        m = results[key]["m"]
        n = results[key]["n"]
        
        # Times
        if results[key].get("assemble_times"):
            avg_assemble = np.mean(results[key]["assemble_times"])
            assemble_df.loc[m, n] = avg_assemble
            
        if results[key].get("our_times"):
            avg_our = np.mean(results[key]["our_times"])
            our_df.loc[m, n] = avg_our
            
            if not pd.isna(assemble_df.loc[m, n]):
                total_our_df.loc[m, n] = avg_our + assemble_df.loc[m, n]
                
        if results[key].get("nodiag_times"):
            avg_nodiag = np.mean(results[key]["nodiag_times"])
            nodiag_df.loc[m, n] = avg_nodiag
            
            if not pd.isna(assemble_df.loc[m, n]):
                total_nodiag_df.loc[m, n] = avg_nodiag + assemble_df.loc[m, n]
        
        # Results (take the last result for each combination)
        if results[key].get("our_results"):
            our_results_df.loc[m, n] = results[key]["our_results"][-1]
            
        if results[key].get("nodiag_results"):
            nodiag_results_df.loc[m, n] = results[key]["nodiag_results"][-1]
    
    # Save dataframes to CSV
    assemble_df.to_csv(os.path.join(output_dir, "assemble_times.csv"))
    our_df.to_csv(os.path.join(output_dir, "our_times.csv"))
    nodiag_df.to_csv(os.path.join(output_dir, "nodiag_times.csv"))
    total_our_df.to_csv(os.path.join(output_dir, "total_our_times.csv"))
    total_nodiag_df.to_csv(os.path.join(output_dir, "total_nodiag_times.csv"))
    our_results_df.to_csv(os.path.join(output_dir, "our_results.csv"))
    nodiag_results_df.to_csv(os.path.join(output_dir, "nodiag_results.csv"))
    
    # Create pretty tables
    tables = {
        "Assembly Times": assemble_df,
        "Our Method Times": our_df,
        "No-Diagonal Method Times": nodiag_df,
        "Total Our Method Times": total_our_df,
        "Total No-Diagonal Method Times": total_nodiag_df
    }
    
    for table_name, df in tables.items():
        pretty_table = df.copy()
        
        # Format times as strings with 4 decimal places
        for m in m_values:
            for n in n_values:
                if not pd.isna(df.loc[m, n]):
                    pretty_table.loc[m, n] = f"{df.loc[m, n]:.4f}"
        
        # Save table to text file
        with open(os.path.join(output_dir, f"{table_name.lower().replace(' ', '_')}.txt"), "w") as f:
            f.write(f"{table_name} (seconds)\n")
            f.write("=" * (len(table_name) + 10) + "\n\n")
            f.write("m\\n  " + "  ".join([f"{n:6d}" for n in n_values]) + "\n")
            for m in m_values:
                row = f"{m:3d}  "
                for n in n_values:
                    val = pretty_table.loc[m, n]
                    row += f"{val:8s}" if not pd.isna(val) else "    N/A  "
                f.write(row + "\n")
    
    # Create plots
    plt.figure(figsize=(12, 8))
    
    # Plot assembly times
    for n in n_values:
        times = [assemble_df.loc[m, n] for m in m_values if not pd.isna(assemble_df.loc[m, n])]
        valid_m = [m for m in m_values if not pd.isna(assemble_df.loc[m, n])]
        if times:
            plt.plot(valid_m, times, marker='o', label=f"n={n} (Assembly)")
    
    plt.xlabel('m (Order)')
    plt.ylabel('Time (seconds)')
    plt.title('U-Statistics Assembly Time')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Use log scale for better visualization
    plt.savefig(os.path.join(output_dir, "assembly_time_plot.png"), dpi=300)
    plt.close()
    
    # Plot computation times
    plt.figure(figsize=(12, 8))
    
    # Our method times
    for n in n_values:
        times = [our_df.loc[m, n] for m in m_values if not pd.isna(our_df.loc[m, n])]
        valid_m = [m for m in m_values if not pd.isna(our_df.loc[m, n])]
        if times:
            plt.plot(valid_m, times, marker='o', label=f"n={n} (Our)")
    
    # No-diagonal method times
    for n in n_values:
        times = [nodiag_df.loc[m, n] for m in m_values if not pd.isna(nodiag_df.loc[m, n])]
        valid_m = [m for m in m_values if not pd.isna(nodiag_df.loc[m, n])]
        if times:
            plt.plot(valid_m, times, marker='x', linestyle='--', label=f"n={n} (No-Diag)")
    
    plt.xlabel('m (Order)')
    plt.ylabel('Time (seconds)')
    plt.title('U-Statistics Computation Time')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Use log scale for better visualization
    plt.savefig(os.path.join(output_dir, "computation_time_plot.png"), dpi=300)
    plt.close()
    
    # Plot total times
    plt.figure(figsize=(12, 8))
    
    # Our method total times
    for n in n_values:
        times = [total_our_df.loc[m, n] for m in m_values if not pd.isna(total_our_df.loc[m, n])]
        valid_m = [m for m in m_values if not pd.isna(total_our_df.loc[m, n])]
        if times:
            plt.plot(valid_m, times, marker='o', label=f"n={n} (Our Total)")
    
    # No-diagonal method total times
    for n in n_values:
        times = [total_nodiag_df.loc[m, n] for m in m_values if not pd.isna(total_nodiag_df.loc[m, n])]
        valid_m = [m for m in m_values if not pd.isna(total_nodiag_df.loc[m, n])]
        if times:
            plt.plot(valid_m, times, marker='x', linestyle='--', label=f"n={n} (No-Diag Total)")
    
    plt.xlabel('m (Order)')
    plt.ylabel('Time (seconds)')
    plt.title('U-Statistics Total Time (Assembly + Computation)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Use log scale for better visualization
    plt.savefig(os.path.join(output_dir, "total_time_plot.png"), dpi=300)
    plt.close()
    
    return tables


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")

    # Configuration
    m_values = [4, 5, 6, 7]  # Order values to test
    n_values = [1000, 2000, 3000, 4000, 5000]  # Sample size values to test
    repeats = 10  # Number of repetitions for each configuration
    summor = "torch"  # Summation method to use
    results_dir="benchmark_results_1"
    path_method = "greedy_minor"  # Path method to use
    
    print(f"Running benchmarks for:")
    print(f"  m values: {m_values}")
    print(f"  n values: {n_values}")
    print(f"  repeats: {repeats}")
    print(f"  summor: {summor}")
    
    # Run benchmarks
    results = run_benchmark(
        m_values=m_values,
        n_values=n_values,
        repeats=repeats,
        summor=summor,
        results_dir=results_dir,
        path_method=path_method,
        run_our=False,  # Set to False to skip test_our method
        run_nodiag=True  # Set to False to skip test_nodiag method
    )
    
    # Analyze results
    analyze_results(results, output_dir=results_dir)
    
    print("Benchmark completed!")
    print("Results saved in ", results_dir, " directory.")
    