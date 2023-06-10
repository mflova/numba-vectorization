from numba import float64, guvectorize

from numba_vectorization.utils.type_aliases import FloatArray


@guvectorize([(float64[:], float64[:])], "(n) -> (n)", target="cpu")
def _guvectorize_cpu_simple(arr: FloatArray, res: FloatArray):
    for i, value in enumerate(arr):
        res[i] = value * i


@guvectorize([(float64[:], float64[:])], "(n) -> (n)", target="parallel")
def _guvectorize_parallel_simple(arr: FloatArray, res: FloatArray):
    for i, value in enumerate(arr):
        res[i] = value * i


def guvec_parallel_simple(arr: FloatArray) -> FloatArray:
    """Perform simple function with guvectorize in parallel."""
    return _guvectorize_parallel_simple(arr)


def guvec_cpu_simple(arr: FloatArray) -> FloatArray:
    """Perform simple function with guvectorize."""
    return _guvectorize_cpu_simple(arr)
