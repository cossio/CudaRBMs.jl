module CudaRBMs

using CUDA: cu
using RestrictedBoltzmannMachines: RBM, ∂RBM,
    Binary, Spin, Potts, Gaussian, ReLU, dReLU, pReLU, xReLU
using Adapt: adapt

gpu(x::AbstractArray) = cu(x)
cpu(x::AbstractArray) = adapt(Array, x)

include("rbm.jl")
include("layers.jl")

end # module
