### some useful functions
from collections import defaultdict
import string, re
import numpy as np
import threading
from queue import Queue
import queue
import logging

AB_talbe = list(string.ascii_lowercase)


# This function is to generate the reverse mapping of a dictionary
def reverse_mapping(mapping: dict) -> dict:
    reverse = defaultdict(list)
    for key, value in mapping.items():
        reverse[value].append(key)
    return reverse


# This functio is to standardize the indexes to continuous integer indexes
def standardize_indexes(lst: list) -> tuple:
    """
    Standardize the index list to continuous integer indexes.

    Args:
        lst (list): List of index pairs. Each pair is a list of some integers.

    Returns:
        tuple: Standardized index list and mapping from standardized to original indexes.
    """
    num_to_index = {}
    standardized_lst = []
    standardized_to_original = {}
    current_index = 0
    for t in lst:
        standardized_pair = []
        for num in t:
            if num not in num_to_index:
                num_to_index[num] = current_index
                standardized_to_original[current_index] = num
                current_index += 1
            standardized_pair.append(num_to_index[num])
        standardized_lst.append(standardized_pair)

    return standardized_lst, standardized_to_original


def numbers_to_letters(numbers: list[int] | list[list]) -> list:
    if isinstance(numbers[0], int):
        return [AB_talbe[num] for num in numbers]
    elif isinstance(numbers[0], list):
        return ["".join([AB_talbe[num] for num in pair]) for pair in numbers]
    else:
        raise TypeError(
            "Input must be a list of integers or a list of lists of integers."
        )


def dedup(strings: str | list) -> str:
    """
    Remove the duplicated characters from a string or a list of strings.

    Args:
        strings (str | list): The input string or a list of strings.

    Returns:
        str: A string without duplicated characters.
             If the input is a list, the result is the concatenation of all strings after deduplication.

    Examples:
        >>> dedup("hello")
        "helo"
        >>> dedup(["hello", "world"])
        "hellowrd"
        >>> dedup("")
        ""
        >>> dedup([])
        ""
    """
    if isinstance(strings, str):
        return "".join(dict.fromkeys(strings))
    elif isinstance(strings, list):
        combined_string = "".join(strings)
        return "".join(dict.fromkeys(combined_string))
    else:
        raise TypeError("Input must be a string or a list of strings.")


def strings2format(lhs_lst: list, rhs: str = None) -> str:
    if rhs is None:
        rhs = ""
    return "->".join([",".join(lhs_lst), rhs])


def analyze_path(compute_format: str, optimize=False, tensor_dim=10, path_func=None):
    if path_func is None:
        path_func = np.einsum_path
    elif path_func == "oe" or path_func == "opt_einsum":
        import opt_einsum

        path_func = opt_einsum.contract_path
    else:
        raise ValueError("Invalid path evaluate function.")
    lhs, result = compute_format.split("->")
    lhs_lst = lhs.split(",")
    compute_format = strings2format(lhs_lst, result)
    ndim_lst = [len(pair) for pair in lhs_lst]
    tensor_list = [np.random.rand(*(tensor_dim,) * ndim) for ndim in ndim_lst]
    path_info = str(path_func(compute_format, *tensor_list, optimize=optimize)[1])
    naive_flop_match = re.search(r"Naive FLOP count:\s*([\d\.]+e[+-]?\d*)", path_info)
    optimized_flop_match = re.search(
        r"Optimized FLOP count:\s*([\d\.]+e[+-]?\d*)", path_info
    )
    largest_intermediate_match = re.search(
        r"Largest intermediate:\s*([\d\.]+e[+-]?\d*)", path_info
    )

    naive_flop = naive_flop_match.group(1) if naive_flop_match else None
    optimized_flop = optimized_flop_match.group(1) if optimized_flop_match else None
    largest_intermediate = (
        largest_intermediate_match.group(1) if largest_intermediate_match else None
    )

    return float(naive_flop), float(optimized_flop), float(largest_intermediate)


def estimate_cpolynomial_degree(n_lst: list, flops: list):
    n_lst = np.array(n_lst)
    flops = np.array(flops)
    log_n = np.log(n_lst)
    log_flops = np.log(flops)
    A = np.vstack([log_n, np.ones(len(log_n))]).T
    m, _ = np.linalg.lstsq(A, log_flops, rcond=None)[0]
    return m


class SequenceWriter:
    """
    A thread-safe class for asynchronously writing path data to an HDF5 file.

    This class uses a producer-consumer model, where the main thread (producer) adds
    data to a queue, and a background thread (consumer) writes the data to the file.

    Attributes:
        queue (Queue): A thread-safe queue for storing data to be written.
        h5_path (str): The path to the HDF5 file where data will be saved.
        running (bool): A flag to control the background thread's execution.
        thread (Thread): The background thread responsible for writing data.
    """

    def __init__(self, h5_path, max_queue_size=1000):
        """
        Initializes the PathWriter instance.

        Args:
            h5_path (str): The path to the HDF5 file where data will be saved.
            max_queue_size (int, optional): The maximum size of the queue. Defaults to 1000.
        """
        self.queue = Queue(maxsize=max_queue_size)
        self.h5_path = h5_path
        self.running = True

        # Start the background thread
        self.thread = threading.Thread(target=self._write_worker, daemon=True)
        self.thread.start()

    def _write_worker(self):
        """
        The worker method for the background thread.

        Continuously retrieves data from the queue and writes it to the HDF5 file.
        Exits when `self.running` is False and the queue is empty.
        """
        while self.running or not self.queue.empty():
            try:
                data = self.queue.get(timeout=1)  # Wait for 1 second
                if data is not None:
                    group_path, path = data
                    path.save(self.h5_path, group_path)
                self.queue.task_done()
            except queue.Empty:
                continue  # Queue is empty, continue waiting
            except Exception as e:
                logging.error(f"Error while writing data: {e}")
                continue

    def add_path(self, path, group_path):
        """
        Adds a path and its group path to the queue for writing.

        Args:
            path (object): The path object to be saved. Must have a `save` method.
            group_path (str): The group path within the HDF5 file where the data will be saved.

        Raises:
            ValueError: If `path` or `group_path` is None.
        """
        if path is None or group_path is None:
            raise ValueError("path and group_path cannot be None")
        self.queue.put((path, group_path))

    def stop(self):
        """
        Stops the background thread gracefully.

        Waits for all remaining tasks in the queue to be processed before stopping the thread.
        """
        self.running = False
        self.queue.join()  # Wait for all tasks in the queue to complete
        self.thread.join()  # Wait for the thread to exit

    def is_running(self):
        """
        Checks if the background thread is still running.

        Returns:
            bool: True if the thread is running, False otherwise.
        """
        return self.thread.is_alive()
