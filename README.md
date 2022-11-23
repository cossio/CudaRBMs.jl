# CudaRBMs Julia package

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/cossio/CudaRBMs.jl/blob/master/LICENSE.md)
![](https://github.com/cossio/CudaRBMs.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/cossio/CudaRBMs.jl/branch/master/graph/badge.svg?token=O5P8LQTVF3)](https://codecov.io/gh/cossio/CudaRBMs.jl)

Some utilities to use [`RestrictedBoltzmannMachines`](https://github.com/cossio/RestrictedBoltzmannMachines.jl) package with `CUDA`.

## Installation

This package is not registered. Install with:

```julia
import Pkg
Pkg.add("git@github.com:cossio/CudaRBMs.jl.git")
```

## Usage

This defines two functions, `cpu` and `gpu` (similar to Flux.jl), to move `RBM` and layers to/from the CPU and GPU.

```julia
using RestrictedBoltzmannMachines: BinaryRBM
using CudaRBMs: cpu, gpu

rbm = BinaryRBM(randn(5), randn(3), randn(5,3)) # in CPU

# copy to GPU
rbm_cu = gpu(rbm)

# ... do some things with rbm_cu on the GPU (e.g. training, sampling)

# copy back to CPU
rbm = cpu(rbm_cuda)
```