from collections import defaultdict
from typing import (
    List,
    Tuple,
    TypeVar,
    Union,
    Callable,
    Optional,
    Any,
    Dict,
    Set,
    Hashable,
)
import string
import threading
from queue import Queue
import queue
import logging
import time
from contextlib import ContextDecorator
from functools import wraps

NestedHashableList = List[Union[str, List[Hashable]]]

_English_alphabet = string.ascii_lowercase + string.ascii_uppercase


# similar to opt_einsum
class Alphabet:
    _instance = None
    _initialized = False

    def __new__(cls) -> "Alphabet":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not Alphabet._initialized:
            self._alphabet = _English_alphabet
            Alphabet._initialized = True

    def __getitem__(self, i: int) -> str:
        if i < 52:
            return self._alphabet[i]
        elif i >= 55296:
            return chr(i + 2048)
        else:
            return chr(i + 140)


AB_table: "Alphabet" = Alphabet()


# This function is to generate the reverse mapping of a dictionary
def reverse_mapping(mapping: dict) -> dict:
    """Generate a reverse mapping from a dictionary where values map to lists
    of keys.

    Args:
        mapping (dict): Input dictionary to be reversed

    Returns:
        dict: Reversed dictionary where each value maps to a list of original keys

    Example:
        >>> d = {'a': 1, 'b': 2, 'c': 1}
        >>> reverse_mapping(d)
        {1: ['a', 'c'], 2: ['b']}
    """
    reverse = defaultdict(list)
    for key, value in mapping.items():
        reverse[value].append(key)
    return dict(reverse)


# This functio is to standardize the indexes to continuous integer indexes
def standardize_indexes(lst: list) -> List[List[int]]:
    """Standardize the index list to continuous integer indexes.

    Args:
        lst (list): List of index pairs. Each pair is a list of some integers.

    Returns:
        List[List[int]]: Standardized index list.
    """
    num_to_index = {}
    standardized_lst = []
    current_index = 0
    for t in lst:
        standardized_pair = []
        for num in t:
            if num not in num_to_index:
                num_to_index[num] = current_index
                current_index += 1
            standardized_pair.append(num_to_index[num])
        standardized_lst.append(standardized_pair)
    return standardized_lst


def numbers_to_letters(numbers: List[int] | List[List[int]]) -> Tuple[List[str], dict]:
    """Convert numbers or lists of numbers to corresponding letters or letter
    combinations, and return a mapping from letters back to numbers.

    Args:
        numbers (List[int] | List[List[int]]):
            List of integers or list of integer lists
            Each integer is converted to corresponding
            lowercase letter (0->a, 1->b, etc)

    Returns:
        Tuple[List[str], dict]: A tuple containing:
            - List of converted letters or letter combinations
            - Dictionary mapping letters back to their original numbers

    Raises:
        TypeError: If input is not a list of integers or list of integer lists
        ValueError: If any number exceeds alphabet size (26)

    Example:
        >>> letters, mapping = numbers_to_letters([0, 1, 2])
        >>> letters
        ['a', 'b', 'c']
        >>> mapping
        {'a': 0, 'b': 1, 'c': 2}
        >>> letters, mapping = numbers_to_letters([[0, 1], [2, 3]])
        >>> letters
        ['ab', 'cd']
        >>> mapping
        {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    """
    try:
        result = []
        letter_to_number = {}

        for lst in numbers:
            if isinstance(lst, int):
                letter = AB_table[lst]
                result.append(letter)
                letter_to_number[letter] = lst
            elif isinstance(lst, list | tuple):
                combined = ""
                for num in lst:
                    letter = AB_table[num]
                    combined += letter
                    letter_to_number[letter] = num
                result.append(combined)
            else:
                raise TypeError(
                    "Input must be a list of integers or list of integer lists"
                )
        return result, letter_to_number
    except IndexError as e:
        raise ValueError(e)


def dedup(strings: str | list) -> str:
    """Remove the duplicated characters from a string or a list of strings.

    Args:
        strings (str | list): The input string or a list of strings.

    Returns:
        str: A string without duplicated characters.
             If the input is a list, the result is the concatenation of
             all strings after deduplication.

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


def strings2format(lhs_lst: list, rhs: Optional[str] = None) -> str:
    """Convert left-hand side list and right-hand side string to arrow format
    notation.

    Args:
        lhs_lst (list): List of strings representing left-hand side operands
        rhs (str, optional): Right-hand side result string. Defaults to empty string

    Returns:
        str: Formatted string in the form "op1,op2,...->result"

    Example:
        >>> strings2format(['ab', 'bc'], 'ac')
        'ab,bc->ac'
        >>> strings2format(['ab', 'bc'])
        'ab,bc->'
    """
    if rhs is None:
        rhs = ""
    return "->".join([",".join(lhs_lst), rhs])


class SequenceWriter:
    """A thread-safe class for asynchronously writing objects to an HDF5 file.

    This class uses a producer-consumer model, where the main thread (producer) adds
    data to a queue, and a background thread (consumer) writes the data to the file.

    Attributes:
        queue (Queue): A thread-safe queue for storing data to be written.
        h5_path (str): The path to the HDF5 file where data will be saved.
        running (bool): A flag to control the background thread's execution.
        thread (Thread): The background thread responsible for writing data.
    """

    def __init__(self, h5_path, max_queue_size=10000) -> None:
        """Initializes the SequenceWriter instance.

        Args:
            h5_path (str):
                The path to the HDF5 file where data will be saved.
            max_queue_size (int, optional):
                The maximum size of the queue. Defaults to 10000.
        """
        self.queue: Queue = Queue(maxsize=max_queue_size)
        self.h5_path = h5_path
        self.running = True

        # Start the background thread
        self.thread = threading.Thread(target=self._write_worker, daemon=True)
        self.thread.start()

    def _write_worker(self) -> None:
        """The worker method for the background thread.

        Continuously retrieves data from the queue and writes it to the HDF5 file.
        Exits when `self.running` is False and the queue is empty.
        """
        while self.running or not self.queue.empty():
            try:
                data = self.queue.get(timeout=1)  # Wait for 1 second
                if data is not None:
                    obj, group_path = data
                    obj.save(self.h5_path, group_path)
                self.queue.task_done()
            except queue.Empty:
                continue  # Queue is empty, continue waiting
            except Exception as e:
                logging.error(f"Error while writing data: {e}")
                continue

    def add_obj(self, obj: Any, group_path=None) -> None:
        """Adds an object and its group path to the queue for writing.

        Args:
            obj (object): The object to be saved. Must have a `save` method.
            group_path (str):
                The group path within the HDF5 file where the data will be saved.

        Raises:
            ValueError: If `obj` is None.
        """
        if obj is None:
            raise ValueError("obj and group_path cannot be None")
        self.queue.put((obj, group_path))

    def stop(self) -> None:
        """Stops the background thread gracefully.

        Waits for all remaining tasks in the queue to be processed before
        stopping the thread.
        """
        self.running = False
        self.queue.join()  # Wait for all tasks in the queue to complete
        self.thread.join()  # Wait for the thread to exit

    def is_running(self) -> bool:
        """Checks if the background thread is still running.

        Returns:
            bool: True if the thread is running, False otherwise.
        """
        return self.thread.is_alive()


def einsum_expression_to_mode(expression: str) -> Tuple[List[str], str]:
    """Convert an Einstein summation expression to a mode string.

    Args:
        expression (str): The Einstein summation expression (e.g., 'ij,jk->ik').

    Returns:
        Tuple[str, str]: A tuple containing the left-hand side
            and right-hand side modes.
    """
    lhs, rhs = expression.split("->")
    lhs_modes = lhs.split(",")
    return lhs_modes, rhs


def get_adj_list(mode: NestedHashableList) -> Dict[int, Set[int]]:
    """Convert a mode list to an adjacency list representation.

    Args:
        mode (NestedHashableList): The mode list to convert.

    Returns:
        Dict[int, Set[int]]: The adjacency list representation of the mode.
    """
    vertices = set()
    for pair in mode:
        vertices.update(set(pair))
    adj_list: Dict = dict()
    for index in vertices:
        adj_list[index] = set()
        for pair in mode:
            if index in pair:
                adj_list[index].update(pair)
        adj_list[index].discard(index)
    return adj_list


_F = TypeVar("_F", bound=Callable[..., Any])


class Timer(ContextDecorator):
    def __init__(
        self, name: Optional[str] = None, logger: Optional[Callable] = print
    ) -> None:
        self.name = name or "Task"
        self.logger = logger
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None) -> None:
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        self.logger(f"{self.name} using: {self.elapsed:.3f} seconds")

    def __call__(self, func: _F) -> _F:
        if self.name is None:
            self.name = func.__name__

        @wraps(func)
        def wrapped(*args, **kwargs) -> Any:
            with self:
                return func(*args, **kwargs)

        return wrapped  # type: ignore


def _initialize_torch():
    try:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch, device
    except ImportError:
        raise ImportError(
            "Torch is not installed. Please install it to use tensor contraction with torch."
        )
