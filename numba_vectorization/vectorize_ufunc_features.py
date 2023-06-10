"""Script that shows advantages and features brough by numpy ufuncs."""
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike

from numba_vectorization.utils.type_aliases import FloatArray
from numba_vectorization.vectorize.core_functions import vectorize_based


def print_arr(arr: ArrayLike, title: str, *, indent: int = 0, end: str = "") -> None:
    """
    Print the array with a specific format.

    Args:
        arr (ArrayLike): Array to print.
        title (str): Title that will preceed the array.
        indent (int, optional): Number of indents used (2-spaced). Defaults to 0.
        end (str, optional): Printed after the array. Defaults to "".
    """
    indents = "  " * indent
    print(f"{indents}{title}")
    for row in arr:
        if isinstance(row, Iterable):
            print(f"{indents}{row}")
        else:
            print(f"{indents}[{row}]")
    print(end)


def test_greater_dimensions(a: FloatArray, b: FloatArray):
    """Test how ufuncs can work on greater dimensions."""
    print("--- Greater dimensiones ---")
    print_arr(a, "First input array", indent=1)
    print_arr(b, "Second input array", indent=1)
    c = vectorize_based.vec_sum(a, b)
    print_arr(c, "Output array", indent=1, end="\n")


def test_reductions(a: FloatArray):
    """
    Test how ufuncs can work by reduction.

    This technique implies that, instead of getting both input values from
    different arrays, these will be got from the a single input array.
    """
    print("--- Reduction feature ---")
    print("Reduction with axis=0")
    print_arr(a, "Input array", indent=1)
    c = vectorize_based.vec_sum.reduce(a, axis=0)
    print_arr(c, "Output array", indent=1, end="\n")

    print("Reduction with axis=1")
    print_arr(a, "Input array", indent=1)
    c = vectorize_based.vec_sum.reduce(a, axis=1)
    print_arr(c, "Output array", indent=1, end="\n")


def test_accumulate(a: FloatArray) -> None:
    """
    Test how ufuncs can work accumulating.

    Similar to `reduction`-based techniques, but this time the output is
    accumulated and propagate to next elements.
    """
    print("--- Accumulate feature ---")
    print("Reduction with axis=0")
    print_arr(a, "Input array", indent=1)
    c = vectorize_based.vec_sum.accumulate(a, axis=1)
    print_arr(c, "Output array", indent=1, end="\n")


def test_mask(a: FloatArray) -> None:
    """
    Test how ufuncs can apply the operation to specific indeces.

    Note how this operations is performed in place. Meaning that your
    input array will be modified.
    """
    print("--- Masking ---")
    a = a.flatten()[:4]
    print_arr(a, "Input array", indent=1)
    indices = [0, 1]
    print_arr(indices, "Applying to indices", indent=1)
    values_to_add = np.array([10, 20])
    print_arr(values_to_add, "The following values", indent=1)
    vectorize_based.vec_sum.at(a, indices, values_to_add)
    print_arr(a, "Output array", indent=1, end="\n")


def main() -> None:
    shape = (3, 4)
    a = np.random.randint(0, 5, shape)
    b = np.random.randint(0, 5, shape)
    test_greater_dimensions(a, b)
    test_reductions(a)
    test_accumulate(a)
    test_mask(a)


if __name__ == "__main__":
    main()
