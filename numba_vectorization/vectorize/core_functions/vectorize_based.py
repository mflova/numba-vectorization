"""All functions whose core are vectorized based."""
import math

import numpy as np
from numba import float64, vectorize

from numba_vectorization.utils.type_aliases import FloatArray


@vectorize([float64(float64, float64)])
def _vec_cpu_func(a: float, b: float) -> float:
    return np.sqrt(a**2 + b**2)


@vectorize([float64(float64, float64)], target="parallel")
def _vec_parallel_func(a: float, b: float) -> float:
    return np.sqrt(a**2 + b**2)


@vectorize([float64(float64, float64)], target="cuda")
def _vec_cuda_func(a: float, b: float) -> float:
    return math.sqrt(a**2 + b**2)


def vec_cpu_func(a: FloatArray, b: FloatArray) -> FloatArray:
    """Vectorized function with numba."""
    return _vec_cpu_func(a, b)


def vec_parallel_func(a: FloatArray, b: FloatArray) -> FloatArray:
    """Vectorized function with numba running on parallel mode."""
    return _vec_parallel_func(a, b)


def vec_cuda_func(a: FloatArray, b: FloatArray) -> FloatArray:
    """Vectorized function with numba running on cuda (GPU) mode."""
    return _vec_cuda_func(a, b)


def vec_np_func(a: FloatArray, b: FloatArray) -> FloatArray:
    """Vectorized function using purely numpy."""
    return np.sqrt(a**2 + b**2)
