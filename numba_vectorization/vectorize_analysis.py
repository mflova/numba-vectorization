"""Perform time-measuring experiment for vectorized functions."""
import argparse
from typing import List, Tuple

from numba_vectorization.utils.utils import Experiment
from numba_vectorization.vectorize.core_functions import scalar_based, vectorize_based


def parse_arguments() -> argparse.Namespace:
    """Parse all input arguments given to the following script."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print-metrics",
        action="store_true",
        help="Set this flag to true to print all resulting metrics.",
    )
    parser.add_argument(
        "--plot-metrics",
        action="store_true",
        help="Set this flag to plot all the resulting metrics.",
    )
    parser.add_argument(
        "--n-calls",
        type=int,
        default=3,
        help="Choose how many times each function will be executed.",
    )
    parser.add_argument(
        "--up-to-n-samples",
        type=int,
        help=(
            "Indicate what's the maximum number of samples that you are willing to "
            "process."
        ),
        default=1_000_000,
    )
    parser.add_argument(
        "--increase-by",
        type=int,
        help=(
            "Number of elements per array will be increased using a geometric series "
            "with this rate until it reaches `up-to-n-samples`."
        ),
        default=5,
    )

    return parser.parse_args()


def generate_shapes(*, increase_by: int, up_to: int) -> Tuple[Tuple[int, ...], ...]:
    """
    Generate multiple shapes by using a geometric series.

    Args:
        increase_by (int): Rate of increase wrt the previous shape.
        up_to (int): Limit of the last shape.

    Returns:
        Tuple[Tuple[int, ...]]: Tuple with multiple shapes.
    """
    init = 1
    shapes: List[Tuple[int, ...]] = []
    i = 1
    while True:
        shape = init * increase_by ** (i - 1)
        if shape >= up_to:
            break
        i += 1
        shapes.append((shape,))
    return tuple(shapes)


def main() -> None:
    # Parse arguments
    args = parse_arguments()
    print_metrics = args.print_metrics
    plot_metrics = args.plot_metrics
    up_to_n_samples = args.up_to_n_samples
    increase_by = args.increase_by

    # Shapes that will be used to generate input data to the functions to test
    shapes = generate_shapes(increase_by=increase_by, up_to=up_to_n_samples)

    # Functions to test
    funcs = (
        scalar_based.for_loop_func,
        scalar_based.for_loop_jit_func,
        vectorize_based.vec_np_func,
        vectorize_based.vec_cpu_func,
        vectorize_based.vec_parallel_func,
        vectorize_based.vec_cuda_func,
    )

    Experiment(funcs, shapes=shapes).run(
        print_metrics=print_metrics, plot_metrics=plot_metrics, silence_warnings=True
    )


if __name__ == "__main__":
    main()
