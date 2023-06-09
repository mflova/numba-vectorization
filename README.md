# numba-vectorization

This repository aims to provide different analyses about the functionalities that numba
vectorize brings. This typically bring extra plots, metrics and parametrization.

## Setup

Before running anything, install the package with:

```bash
pip install .
```

This will guarantee that all dependencies are solved.

## Current implemented tools:

- `numba_vectorization\vectorize_analysis.py`. Arguments:
  - `--print-metrics`: `True` to print all resulting metrics.
  - `--plot-metrics`: `True` to plot all metrics based on number of samples used.
  - `--n-calls`: Choose how many times each function is evaluated to perform its average.
  - `--up-to-n-samples`: Indicates maximum number of samples to be processed.
  - `--increase-by`: Rate of increase in the sample size used in each iteration.