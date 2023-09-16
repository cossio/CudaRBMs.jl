module CudaRBMs

import cuDNN
using CUDA: cu
using Adapt: adapt
using RestrictedBoltzmannMachines: RBM, ∂RBM,
    Binary, Spin, Potts, Gaussian, ReLU, dReLU, pReLU, xReLU
using StandardizedRestrictedBoltzmannMachines: StandardizedRBM
using CenteredRBMs: CenteredRBM

gpu(x::AbstractArray) = cu(x)
cpu(x::AbstractArray) = adapt(Array, x)

include("layers.jl")
include("rbm.jl")
include("std.jl")
include("center.jl")

end # module
