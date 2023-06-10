"""Utils that can be shared among different scripts or modules."""
import inspect
import timeit
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Callable,
    DefaultDict,
    Iterable,
    List,
    MutableMapping,
    Sequence,
    Set,
    Tuple,
)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from numba import NumbaPerformanceWarning

from numba_vectorization.utils.type_aliases import FloatArray


@dataclass(frozen=True)
class Experiment:
    """Perform an experiment to measure time executions of multiple functions."""

    funcs: Iterable[Callable]
    """Functions that will be launched."""
    shapes: Tuple[Tuple[int, ...], ...]
    """Shapes that will be used to create the input arrays to functions."""
    n_calls_per_function: int = 3
    """Used to compute the average execution time for each function."""

    _longer_func_name: int = field(init=False, default=0)
    """Maximum length found among all function to be tested."""
    _n_input_params: int = field(init=False, default=0)
    """Number of input parameters that our functions require."""

    def __post_init__(self) -> None:
        """Perform sanity checks over the functions to be tested."""
        longer_func_name = 0
        n_params: Set[int] = set()
        for func in self.funcs:
            n_params.add(len(inspect.signature(func).parameters))
            longer_func_name = max(longer_func_name, len(func.__name__))
        if len(n_params) != 1:
            raise ValueError(
                "All functions must receive same number of input parameters."
            )

        # Set frozen attributes
        object.__setattr__(self, "_longer_func_name", longer_func_name)
        object.__setattr__(self, "_n_input_params", list(n_params)[0])

    def run(
        self,
        *,
        silence_warnings: bool = False,
        print_metrics: bool = False,
        plot_metrics: bool = False,
    ) -> None:
        """
        Run the experiment by calling all functions with all shapes given.

        All functions will be called at first to avoid any possible delay due to JIT.

        Args:
            silence_warnings (bool, optional): Set to `True` to silent cuda warning
                related to low performance due to low number of samples.
                Defaults to False.
            print_metrics (bool, optional): Set to `True` to print all metrics. Defaults
                to False.
            plot_metrics (bool, optional): Set to `True` to plot the resulting metrics.
                Defaults to False.
        """
        if silence_warnings:
            warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
        times: DefaultDict[str, List[float]] = defaultdict(list)
        n_samples_used: list[int] = []

        # Iterate over each given shape
        self._pre_run_all_funcs()
        for shape in self.shapes:
            data = self._create_data(shape=shape)
            n_samples_used.append(len(data[0]))
            if print_metrics:
                print(f"\n --- Shape of arrays used: {shape} ---")
            for func in self.funcs:
                t = timeit.timeit(lambda: func(*data), number=self.n_calls_per_function)
                t = t / self.n_calls_per_function
                times[func.__name__].append(t)
                if print_metrics:
                    print(f"{self._func_name_with_spaces(func.__name__)}{t:.5f}s")
        else:
            print("\nFinished")
        if plot_metrics:
            self._plot_metrics(n_samples_used, times)

    def _pre_run_all_funcs(self) -> None:
        """Run all functions in order to perform potential JITs."""
        data = self._create_data(shape=self.shapes[0])
        for func in self.funcs:
            func(*data)

    def _create_data(self, *, shape: tuple[int, ...]) -> tuple[FloatArray, ...]:
        """
        Create the same number of arrays that all functions require.

        Args:
            shape (tuple[int, ...]): Shape that all arrays will have.

        Returns:
            tuple[FloatArray, ...]: All arrays that can be directly fed as input to
                functions to be tested.
        """
        return tuple([np.random.rand(*shape) for _ in range(self._n_input_params)])

    def _func_name_with_spaces(self, func_name: str) -> str:
        """
        Format given function name with multiple spaces for aligning purposes.

        Args:
            func_name (str): Name of the function that needs to be formatted.

        Returns:
            str: Name of the function with spaces after its name.
        """
        spaces = (self._longer_func_name + 2 - len(func_name)) * " "
        return f"{func_name}{spaces}"

    def _plot_metrics(
        self, n_samples_used: List[int], times_data: MutableMapping[str, Sequence[float]]
    ) -> None:
        """
        Plot all metrics obtained.

        Args:
            n_samples_used (List[int]): Number of samples used. It will be the X axis.
            times_data (MutableMapping[str, List[float]]): Dictionary that links the name
                of the function with a sequence contraining all of the time executions for
                the given `n_samples_used`.
        """
        base_func_name = list(times_data.keys())[0]
        base_time = times_data[base_func_name][-1]
        for func_name, times in times_data.items():
            style = "" if "vec" not in func_name else "--"
            speed_up = int(base_time / times[-1])
            plt.plot(
                n_samples_used,
                times,
                style,
                label=f"{self._func_name_with_spaces(func_name)}(x{speed_up} speed up)",
            )

        ax = plt.gca()
        formatter = ticker.FuncFormatter(lambda x, pos: f"{int(x/1000)}k")
        ax.xaxis.set_major_formatter(formatter)

        plt.yscale("log")
        plt.xlabel("Number of samples used")
        plt.ylabel("Exeuction time")
        plt.title(f"Execution times using {self.n_calls_per_function} for each average.")
        plt.grid(True, which="both")

        plt.xlim(0, np.max(n_samples_used))
        plt.legend()
        plt.show()
