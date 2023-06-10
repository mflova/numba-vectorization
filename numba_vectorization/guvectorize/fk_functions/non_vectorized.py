"""
Module that implements all non-vectorized forward kinematics equations.

These are divided into:
    - Private methods: Implement the core functionality
    - Public methods: Act as a wrapper and prepare the data for the private ones. In
      this case, these ones implement the for loop to pass the data.
"""
import numpy as np
from numba import njit

from numba_vectorization.utils.type_aliases import FloatArray


def _for_loop_fk(joints: FloatArray) -> FloatArray:
    q1, q2, q3, q4, q5, q6 = joints

    d1 = 10.0
    a1 = 2.0
    a2 = 3.0
    a3 = 3.0
    d4 = 2.0
    d6 = 5.0

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

    return np.array(
        [[nx, ox, ax, px], [ny, oy, ay, py], [nz, oz, az, pz], [0.0, 0.0, 0.0, 1.0]]
    )


@njit
def _for_loop_njit_fk(joints: FloatArray) -> FloatArray:
    res = []
    for joint_row in joints:
        q1, q2, q3, q4, q5, q6 = joint_row
        d1 = 10.0
        a1 = 2.0
        a2 = 3.0
        a3 = 3.0
        d4 = 2.0
        d6 = 5.0

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
            c1 * s23 * s5
            - c5 * (s1 * c23 * c4 + s1 * s4)
            + c6 * (s1 * c4 - c1 * c23 * s4)
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

        res.append(
            np.array(
                [
                    [nx, ox, ax, px],
                    [ny, oy, ay, py],
                    [nz, oz, az, pz],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )
    return res


def for_loop_fk(arr: FloatArray) -> FloatArray:
    """Perform FK equations with a pure for loop"""
    lst: list[FloatArray] = []
    for row in arr:
        lst.append(_for_loop_fk(row))
    return np.vstack(lst).T


def for_loop_njit_fk(arr: FloatArray) -> FloatArray:
    """Perform FK equations with a for loop and pre-compiled core function with numba."""
    return _for_loop_njit_fk(arr)
