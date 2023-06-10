import numpy as np

from numba_vectorization.utils.type_aliases import FloatArray


def simple(arr: FloatArray) -> FloatArray:
    """Define simple function that multiplies value by its index."""
    out = np.empty(arr.shape)
    for i, _ in enumerate(arr):
        out[i] = arr[i] * i
    return out
