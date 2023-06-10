# numba-vectorization

This repository aims to provide different analyses about the functionalities that numba
vectorize brings. This typically bring extra plots, metrics and parametrization.

## Setup

Before running anything, install the package with:

```bash
pip install .
```

This will guarantee that all dependencies are solved.

## Current implemented tools

### Universal functions with @vectorize

- `numba_vectorization\vectorize_analysis.py`: Perform an analysis that evaluates the
  performance of different techniques with vectorized functions. Arguments:
  - `--print-metrics`: `True` to print all resulting metrics.
  - `--plot-metrics`: `True` to plot all metrics based on number of samples used.
  - `--n-calls`: Choose how many times each function is evaluated to perform its average.
  - `--up-to-n-samples`: Indicates maximum number of samples to be processed.
  - `--increase-by`: Rate of increase in the sample size used in each iteration.
- `numba_vectorization\vectorize_ufunc_features.py`: Check how a vectorized
  function with numba acquire extra features thanks to `numpy ufuncs`.

### Generalized universal functions with @guvectorize

- `numba_vectorization\simple_guvectorize_analysis.py`: Perform an analysis that
  evaluates the performance of different techniques with simple guvectorized function.
  Arguments:
  - `--print-metrics`: `True` to print all resulting metrics.
  - `--plot-metrics`: `True` to plot all metrics based on number of samples used.
  - `--n-calls`: Choose how many times each function is evaluated to perform its average.
  - `--up-to-n-samples`: Indicates maximum number of samples to be processed.
  - `--increase-by`: Rate of increase in the sample size used in each iteration.
- `numba_vectorization\fk_guvectorize_analysis.py`: Perform an analysis that evaluates
  the performance of different techniques with complex guvectorized functions. These
  functions perform forward kinematics equation for an arbitrary robot with 6 joints.
  Arguments:
  - `--print-metrics`: `True` to print all resulting metrics.
  - `--plot-metrics`: `True` to plot all metrics based on number of samples used.
  - `--n-calls`: Choose how many times each function is evaluated to perform its average.
  - `--up-to-n-samples`: Indicates maximum number of samples to be processed.
  - `--increase-by`: Rate of increase in the sample size used in each iteration.