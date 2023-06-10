"""Module containing all user defined type aliases."""
import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

FloatArray: TypeAlias = NDArray[np.float64]
"""Any numpy array with float64 values."""
