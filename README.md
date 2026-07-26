# CudaRBMs Julia package

> [!WARNING]
> This package is deprecated. Its functionality has been incorporated in the [`RestrictedBoltzmannMachines`](https://github.com/cossio/RestrictedBoltzmannMachines.jl) package using the [Extensions](https://pkgdocs.julialang.org/v1/creating-packages/#Conditional-loading-of-code-in-packages-(Extensions)) mechanism.

Some utilities to use [`RestrictedBoltzmannMachines`](https://github.com/cossio/RestrictedBoltzmannMachines.jl) package with `CUDA`.

## Installation

This package is registered. Install with:

```julia
import Pkg
Pkg.add("CudaRBMs")
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