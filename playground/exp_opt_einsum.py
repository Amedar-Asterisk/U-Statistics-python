from init import *
import opt_einsum as oe
import U_stats.tensor_contraction.state as state


def test1():
    n = 100
    rep = "abc,def->"
    shapes = tensers_shape_of_format(rep, n)
    path1 = oe.contract_path(rep, *shapes, optimize="optimal", shapes=True)
    print(path1[1])


def test2():
    rep = ["abc", "def", "ghi"]
    mode = state.TensorContractionState(rep)
    print(mode)
    while i := mode.next_contract():
        _, _, computing_exp = mode.contract(i)
        tensors_shape = tensers_shape_of_format(computing_exp)
        print(computing_exp)
        print(tensors_shape)
        path = oe.contract_path(
            computing_exp, *tensors_shape, optimize=False, shapes=True
        )
        print(path[1])


def test3():
    n = 100
    rep = "abc,def->"
    shapes = tensers_shape_of_format(rep, n)
    tensors = virtual_tensors(shapes)
    path1 = np.einsum_path(rep, *tensors, optimize=False)
    print(path1[1])


def tensers_shape_of_format(format: str, n: int = 100) -> list:
    """
    Convert a string format to a list of tensor shapes.
    :param format: A string format like "abc,def,ghi->"
    :return: A list of tensor shapes.
    """
    input_part, _ = format.split("->")
    return [(n,) * len(inputs) for inputs in input_part.split(",")]


def virtual_tensors(shapes: list) -> list:
    """
    Convert a list of tensor shapes to a list of virtual tensors.
    :param shapes: A list of tensor shapes.
    :return: A list of virtual tensors.
    """
    return [np.random.rand(*shape) for shape in shapes]


class VirtualTensor:
    def __init__(self, shape):
        self.shape = shape
        self.data = None

    def __repr__(self):
        return f"VirtualTensor(shape={self.shape})"


if __name__ == "__main__":
    test1()
