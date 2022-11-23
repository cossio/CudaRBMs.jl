# CudaRBMs Julia package

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/cossio/CudaRBMs.jl/blob/master/LICENSE.md)
![](https://github.com/cossio/CudaRBMs.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/cossio/CudaRBMs.jl/branch/master/graph/badge.svg?token=O5P8LQTVF3)](https://codecov.io/gh/cossio/CudaRBMs.jl)

Some utilities to use `RestrictedBoltzmannMachines`.jl package with `CUDA`.

## Installation

This package is not registered. Install with:

```julia
import Pkg
Pkg.add("git@github.com:cossio/CudaRBMs.jl.git")
```

## Usage

This defines two functions, `cpu` and `gpu` (similar to Flux.jl), to move `RBM` and layers to/from the CPU and GPU.