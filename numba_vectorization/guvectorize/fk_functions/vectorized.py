"""
Implementation of vectorized based functions.

These are divided into:
    - Private methods: Implement the core functionality
    - Public methods: Act as a wrapper and prepare the data for the private ones.
"""
import numpy as np
from numba import bool_, float64, guvectorize, njit

from numba_vectorization.utils.type_aliases import FloatArray


@guvectorize(
    [(float64[:], bool_[:, :], float64[:, :])],
    "(n), (m,m) -> (m,m)",
    nopython=True,
    target="cpu",
)
def _guvec_cpu_fk(joints: FloatArray, dummy, res) -> FloatArray:
    q1, q2, q3, q4, q5, q6 = joints

    d1 = 10
    a1 = 2
    a2 = 3
    a3 = 3
    d4 = 2
    d6 = 5

    c1 = np.cos(q1)
    c2 = np.cos(q2)
    c4 = np.cos(q4)
    c5 = np.cos(q5)
    c6 = np.cos(q6)

    s1 = np.sin(q1)
    s2 = np.sin(q2)
    s4 = np.sin(q4)
    s5 = np.sin(q5)
    s6 = np.sin(q6)

    s23 = np.sin(q3 + q2)
    c23 = np.cos(q2 + q3)

    nx = c6 * (c5 * (c1 * c23 * c4 + s1 * s4) - c1 * s23 * s5) + s6 * (
        s1 * c4 - c1 * c23 * s4
    )
    ny = c6 * (c5 * (s1 * c23 * c4 + c1 * s4) - s1 * s23 * s5) - s6 * (
        c1 * c4 - s1 * c23 * s4
    )
    nz = c6 * (c23 * s5 + s23 * c4 * c5) - s23 * s4 * s6
    ox = s6 * (
        c1 * s23 * s5 - c5 * (s1 * c23 * c4 + s1 * s4) + c6 * (s1 * c4 - c1 * c23 * s4)
    )
    oy = s6 * (s1 * s23 * s5 - c5 * (s1 * c23 * c4 + c1 * s4)) - c6 * (
        c1 * c4 + s1 * c23 * s4
    )
    oz = -s6 * (c23 * s5 + s23 * c4 * c5) - s23 * s4 * c6
    ax = s5 * (c1 * c23 * c4 + s1 * s4) + c1 * s23 * c5
    ay = s5 * (s1 * c23 * c4 - c1 * s4) + s1 * s23 * c5
    az = s23 * c4 * c5 - c23 * c5
    px = d6 * (s5 * (c1 * c23 * c4 + s1 + s4) + c1 * s23 * c5) + c1 * (
        a1 + a2 * c2 + a3 * c23 + d4 * s23
    )
    py = d6 * (s5 * (s1 * c23 * c4 + c1 * s4) + s1 * s23 * c5) + s1 * (
        a1 + a2 * c2 + a3 * c23 + d4 * s23
    )
    pz = a2 * s2 + d1 + a3 * s23 - d4 * c23 + d6 * (s23 * c4 * s5 - c23 * c5)

    res[0, 0] = nx
    res[1, 0] = ny
    res[2, 0] = nz
    res[0, 1] = ox
    res[1, 1] = oy
    res[2, 1] = oz
    res[0, 2] = ax
    res[1, 2] = ay
    res[2, 2] = az
    res[0, 3] = px
    res[1, 3] = py
    res[2, 3] = pz
    res[3, 0] = 0.0
    res[3, 1] = 0.0
    res[3, 2] = 0.0
    res[3, 3] = 1.0


@guvectorize(
    [(float64[:], bool_[:, :], float64[:, :])],
    "(n), (m,m) -> (m,m)",
    nopython=True,
    target="parallel",
)
def _guvec_parallel_fk(joints: FloatArray, dummy, res) -> FloatArray:
    q1, q2, q3, q4, q5, q6 = joints

    d1 = 10
    a1 = 2
    a2 = 3
    a3 = 3
    d4 = 2
    d6 = 5

    c1 = np.cos(q1)
    c2 = np.cos(q2)
    c4 = np.cos(q4)
    c5 = np.cos(q5)
    c6 = np.cos(q6)

    s1 = np.sin(q1)
    s2 = np.sin(q2)
    s4 = np.sin(q4)
    s5 = np.sin(q5)
    s6 = np.sin(q6)

    s23 = np.sin(q3 + q2)
    c23 = np.cos(q2 + q3)

    nx = c6 * (c5 * (c1 * c23 * c4 + s1 * s4) - c1 * s23 * s5) + s6 * (
        s1 * c4 - c1 * c23 * s4
    )
    ny = c6 * (c5 * (s1 * c23 * c4 + c1 * s4) - s1 * s23 * s5) - s6 * (
        c1 * c4 - s1 * c23 * s4
    )
    nz = c6 * (c23 * s5 + s23 * c4 * c5) - s23 * s4 * s6
    ox = s6 * (
        c1 * s23 * s5 - c5 * (s1 * c23 * c4 + s1 * s4) + c6 * (s1 * c4 - c1 * c23 * s4)
    )
    oy = s6 * (s1 * s23 * s5 - c5 * (s1 * c23 * c4 + c1 * s4)) - c6 * (
        c1 * c4 + s1 * c23 * s4
    )
    oz = -s6 * (c23 * s5 + s23 * c4 * c5) - s23 * s4 * c6
    ax = s5 * (c1 * c23 * c4 + s1 * s4) + c1 * s23 * c5
    ay = s5 * (s1 * c23 * c4 - c1 * s4) + s1 * s23 * c5
    az = s23 * c4 * c5 - c23 * c5
    px = d6 * (s5 * (c1 * c23 * c4 + s1 + s4) + c1 * s23 * c5) + c1 * (
        a1 + a2 * c2 + a3 * c23 + d4 * s23
    )
    py = d6 * (s5 * (s1 * c23 * c4 + c1 * s4) + s1 * s23 * c5) + s1 * (
        a1 + a2 * c2 + a3 * c23 + d4 * s23
    )
    pz = a2 * s2 + d1 + a3 * s23 - d4 * c23 + d6 * (s23 * c4 * s5 - c23 * c5)

    res[0, 0] = nx
    res[1, 0] = ny
    res[2, 0] = nz
    res[0, 1] = ox
    res[1, 1] = oy
    res[2, 1] = oz
    res[0, 2] = ax
    res[1, 2] = ay
    res[2, 2] = az
    res[0, 3] = px
    res[1, 3] = py
    res[2, 3] = pz
    res[3, 0] = 0.0
    res[3, 1] = 0.0
    res[3, 2] = 0.0
    res[3, 3] = 1.0


@guvectorize(
    [(float64[:], bool_[:, :], float64[:, :])],
    "(n), (m,m) -> (m,m)",
    nopython=True,
    target="cuda",
)
def _guvec_cuda_fk(joints: FloatArray, dummy, res) -> FloatArray:
    # Matriz de transformación homogénea desde la base hasta la articulación 1

    q1, q2, q3, q4, q5, q6 = joints

    d1 = 10
    a1 = 2
    a2 = 3
    a3 = 3
    d4 = 2
    d6 = 5

    c1 = np.cos(q1)
    c2 = np.cos(q2)
    c4 = np.cos(q4)
    c5 = np.cos(q5)
    c6 = np.cos(q6)

    s1 = np.sin(q1)
    s2 = np.sin(q2)
    s4 = np.sin(q4)
    s5 = np.sin(q5)
    s6 = np.sin(q6)

    s23 = np.sin(q3 + q2)
    c23 = np.cos(q2 + q3)

    nx = c6 * (c5 * (c1 * c23 * c4 + s1 * s4) - c1 * s23 * s5) + s6 * (
        s1 * c4 - c1 * c23 * s4
    )
    ny = c6 * (c5 * (s1 * c23 * c4 + c1 * s4) - s1 * s23 * s5) - s6 * (
        c1 * c4 - s1 * c23 * s4
    )
    nz = c6 * (c23 * s5 + s23 * c4 * c5) - s23 * s4 * s6
    ox = s6 * (
        c1 * s23 * s5 - c5 * (s1 * c23 * c4 + s1 * s4) + c6 * (s1 * c4 - c1 * c23 * s4)
    )
    oy = s6 * (s1 * s23 * s5 - c5 * (s1 * c23 * c4 + c1 * s4)) - c6 * (
        c1 * c4 + s1 * c23 * s4
    )
    oz = -s6 * (c23 * s5 + s23 * c4 * c5) - s23 * s4 * c6
    ax = s5 * (c1 * c23 * c4 + s1 * s4) + c1 * s23 * c5
    ay = s5 * (s1 * c23 * c4 - c1 * s4) + s1 * s23 * c5
    az = s23 * c4 * c5 - c23 * c5
    px = d6 * (s5 * (c1 * c23 * c4 + s1 + s4) + c1 * s23 * c5) + c1 * (
        a1 + a2 * c2 + a3 * c23 + d4 * s23
    )
    py = d6 * (s5 * (s1 * c23 * c4 + c1 * s4) + s1 * s23 * c5) + s1 * (
        a1 + a2 * c2 + a3 * c23 + d4 * s23
    )
    pz = a2 * s2 + d1 + a3 * s23 - d4 * c23 + d6 * (s23 * c4 * s5 - c23 * c5)

    res[0, 0] = nx
    res[1, 0] = ny
    res[2, 0] = nz
    res[0, 1] = ox
    res[1, 1] = oy
    res[2, 1] = oz
    res[0, 2] = ax
    res[1, 2] = ay
    res[2, 2] = az
    res[0, 3] = px
    res[1, 3] = py
    res[2, 3] = pz
    res[3, 0] = 0.0
    res[3, 1] = 0.0
    res[3, 2] = 0.0
    res[3, 3] = 1.0


def _vec_np_fk(joints):
    q1 = joints[:, 0]
    q2 = joints[:, 1]
    q3 = joints[:, 2]
    q4 = joints[:, 3]
    q5 = joints[:, 4]
    q6 = joints[:, 5]

    d1 = 10
    a1 = 2
    a2 = 3
    a3 = 3
    d4 = 2
    d6 = 5

    c1 = np.cos(q1)
    c2 = np.cos(q2)
    c4 = np.cos(q4)
    c5 = np.cos(q5)
    c6 = np.cos(q6)

    s1 = np.sin(q1)
    s2 = np.sin(q2)
    s4 = np.sin(q4)
    s5 = np.sin(q5)
    s6 = np.sin(q6)

    s23 = np.sin(q3 + q2)
    c23 = np.cos(q2 + q3)

    nx = c6 * (c5 * (c1 * c23 * c4 + s1 * s4) - c1 * s23 * s5) + s6 * (
        s1 * c4 - c1 * c23 * s4
    )
    ny = c6 * (c5 * (s1 * c23 * c4 + c1 * s4) - s1 * s23 * s5) - s6 * (
        c1 * c4 - s1 * c23 * s4
    )
    nz = c6 * (c23 * s5 + s23 * c4 * c5) - s23 * s4 * s6
    ox = s6 * (
        c1 * s23 * s5 - c5 * (s1 * c23 * c4 + s1 * s4) + c6 * (s1 * c4 - c1 * c23 * s4)
    )
    oy = s6 * (s1 * s23 * s5 - c5 * (s1 * c23 * c4 + c1 * s4)) - c6 * (
        c1 * c4 + s1 * c23 * s4
    )
    oz = -s6 * (c23 * s5 + s23 * c4 * c5) - s23 * s4 * c6
    ax = s5 * (c1 * c23 * c4 + s1 * s4) + c1 * s23 * c5
    ay = s5 * (s1 * c23 * c4 - c1 * s4) + s1 * s23 * c5
    az = s23 * c4 * c5 - c23 * c5
    px = d6 * (s5 * (c1 * c23 * c4 + s1 + s4) + c1 * s23 * c5) + c1 * (
        a1 + a2 * c2 + a3 * c23 + d4 * s23
    )
    py = d6 * (s5 * (s1 * c23 * c4 + c1 * s4) + s1 * s23 * c5) + s1 * (
        a1 + a2 * c2 + a3 * c23 + d4 * s23
    )
    pz = a2 * s2 + d1 + a3 * s23 - d4 * c23 + d6 * (s23 * c4 * s5 - c23 * c5)

    zeros = np.zeros(len(nx))
    ones = np.zeros(len(nx))
    return np.column_stack(
        [nx, ox, ax, px, ny, oy, ay, py, nz, oz, az, pz, zeros, zeros, zeros, ones]
    )


@njit
def _vec_njit_np_fk(joints):
    q1 = joints[:, 0]
    q2 = joints[:, 1]
    q3 = joints[:, 2]
    q4 = joints[:, 3]
    q5 = joints[:, 4]
    q6 = joints[:, 5]

    d1 = 10
    a1 = 2
    a2 = 3
    a3 = 3
    d4 = 2
    d6 = 5

    c1 = np.cos(q1)
    c2 = np.cos(q2)
    c4 = np.cos(q4)
    c5 = np.cos(q5)
    c6 = np.cos(q6)

    s1 = np.sin(q1)
    s2 = np.sin(q2)
    s4 = np.sin(q4)
    s5 = np.sin(q5)
    s6 = np.sin(q6)

    s23 = np.sin(q3 + q2)
    c23 = np.cos(q2 + q3)

    nx = c6 * (c5 * (c1 * c23 * c4 + s1 * s4) - c1 * s23 * s5) + s6 * (
        s1 * c4 - c1 * c23 * s4
    )
    ny = c6 * (c5 * (s1 * c23 * c4 + c1 * s4) - s1 * s23 * s5) - s6 * (
        c1 * c4 - s1 * c23 * s4
    )
    nz = c6 * (c23 * s5 + s23 * c4 * c5) - s23 * s4 * s6
    ox = s6 * (
        c1 * s23 * s5 - c5 * (s1 * c23 * c4 + s1 * s4) + c6 * (s1 * c4 - c1 * c23 * s4)
    )
    oy = s6 * (s1 * s23 * s5 - c5 * (s1 * c23 * c4 + c1 * s4)) - c6 * (
        c1 * c4 + s1 * c23 * s4
    )
    oz = -s6 * (c23 * s5 + s23 * c4 * c5) - s23 * s4 * c6
    ax = s5 * (c1 * c23 * c4 + s1 * s4) + c1 * s23 * c5
    ay = s5 * (s1 * c23 * c4 - c1 * s4) + s1 * s23 * c5
    az = s23 * c4 * c5 - c23 * c5
    px = d6 * (s5 * (c1 * c23 * c4 + s1 + s4) + c1 * s23 * c5) + c1 * (
        a1 + a2 * c2 + a3 * c23 + d4 * s23
    )
    py = d6 * (s5 * (s1 * c23 * c4 + c1 * s4) + s1 * s23 * c5) + s1 * (
        a1 + a2 * c2 + a3 * c23 + d4 * s23
    )
    pz = a2 * s2 + d1 + a3 * s23 - d4 * c23 + d6 * (s23 * c4 * s5 - c23 * c5)

    zeros = np.zeros(len(nx))
    ones = np.zeros(len(nx))
    return [nx, ox, ax, px, ny, oy, ay, py, nz, oz, az, pz, zeros, zeros, zeros, ones]

@njit(parallel=True)
def _vec_njit_parallel_np_fk(joints):
    q1 = joints[:, 0]
    q2 = joints[:, 1]
    q3 = joints[:, 2]
    q4 = joints[:, 3]
    q5 = joints[:, 4]
    q6 = joints[:, 5]

    d1 = 10
    a1 = 2
    a2 = 3
    a3 = 3
    d4 = 2
    d6 = 5

    c1 = np.cos(q1)
    c2 = np.cos(q2)
    c4 = np.cos(q4)
    c5 = np.cos(q5)
    c6 = np.cos(q6)

    s1 = np.sin(q1)
    s2 = np.sin(q2)
    s4 = np.sin(q4)
    s5 = np.sin(q5)
    s6 = np.sin(q6)

    s23 = np.sin(q3 + q2)
    c23 = np.cos(q2 + q3)

    nx = c6 * (c5 * (c1 * c23 * c4 + s1 * s4) - c1 * s23 * s5) + s6 * (
        s1 * c4 - c1 * c23 * s4
    )
    ny = c6 * (c5 * (s1 * c23 * c4 + c1 * s4) - s1 * s23 * s5) - s6 * (
        c1 * c4 - s1 * c23 * s4
    )
    nz = c6 * (c23 * s5 + s23 * c4 * c5) - s23 * s4 * s6
    ox = s6 * (
        c1 * s23 * s5 - c5 * (s1 * c23 * c4 + s1 * s4) + c6 * (s1 * c4 - c1 * c23 * s4)
    )
    oy = s6 * (s1 * s23 * s5 - c5 * (s1 * c23 * c4 + c1 * s4)) - c6 * (
        c1 * c4 + s1 * c23 * s4
    )
    oz = -s6 * (c23 * s5 + s23 * c4 * c5) - s23 * s4 * c6
    ax = s5 * (c1 * c23 * c4 + s1 * s4) + c1 * s23 * c5
    ay = s5 * (s1 * c23 * c4 - c1 * s4) + s1 * s23 * c5
    az = s23 * c4 * c5 - c23 * c5
    px = d6 * (s5 * (c1 * c23 * c4 + s1 + s4) + c1 * s23 * c5) + c1 * (
        a1 + a2 * c2 + a3 * c23 + d4 * s23
    )
    py = d6 * (s5 * (s1 * c23 * c4 + c1 * s4) + s1 * s23 * c5) + s1 * (
        a1 + a2 * c2 + a3 * c23 + d4 * s23
    )
    pz = a2 * s2 + d1 + a3 * s23 - d4 * c23 + d6 * (s23 * c4 * s5 - c23 * c5)

    zeros = np.zeros(len(nx))
    ones = np.zeros(len(nx))
    return [nx, ox, ax, px, ny, oy, ay, py, nz, oz, az, pz, zeros, zeros, zeros, ones]


def vec_np_fk(arr: FloatArray) -> FloatArray:
    """Perform FK equations with pure numpy vectorized."""
    return _vec_np_fk(arr)


def vec_njit_np_fk(arr: FloatArray) -> FloatArray:
    """Perform FK equations with numpy vectorized pre-compiled with numba."""
    return np.column_stack(_vec_njit_np_fk(arr))

def vec_njit_parallel_np_fk(arr: FloatArray) -> FloatArray:
    """Perform FK equations with numpy vectorized pre-compiled with numba."""
    return np.column_stack(_vec_njit_parallel_np_fk(arr))


def guvec_cpu_fk(arr: FloatArray) -> FloatArray:
    """Perform FK equations using guvectorize from numba."""
    dummy = np.empty((4, 4), dtype=np.bool_)
    return _guvec_cpu_fk(arr, dummy)


def guvec_parallel_fk(arr: FloatArray) -> FloatArray:
    """Perform FK equations using parallel guvectorize from numba."""
    dummy = np.empty((4, 4), dtype=np.bool_)
    return _guvec_parallel_fk(arr, dummy)


def guvec_cuda_fk(arr: FloatArray) -> FloatArray:
    """Perform FK equations using cuda guvectorize from numba."""
    dummy = np.empty((4, 4), dtype=np.bool_)
    return _guvec_cuda_fk(arr, dummy)
