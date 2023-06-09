"""All functions whose core are scalar-based."""
import numpy as np
from numba import float64, njit

from numba_vectorization.utils.type_aliases import FloatArray


def _for_loop_func(a: float, b: float) -> float:
    return np.sqrt(a**2 + b**2)


@njit([float64(float64, float64)])
def _for_loop_jit_func(a: float, b: float) -> float:
    return np.sqrt(a**2 + b**2)


def for_loop_func(a: FloatArray, b: FloatArray) -> FloatArray:
    """For loop based function."""
    results: list[float] = []
    for a_, b_ in zip(a, b):
        results.append(_for_loop_func(a_, b_))
    return np.array(results)


def for_loop_jit_func(a: FloatArray, b: FloatArray) -> FloatArray:
    """For loop based function optimized with numba JIT."""
    results: list[float] = []
    for a_, b_ in zip(a, b):
        results.append(_for_loop_jit_func(a_, b_))
    return np.array(results)
