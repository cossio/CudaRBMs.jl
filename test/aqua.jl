import Aqua
import CudaRBMs

using Test: @testset

@testset "aqua" begin
    Aqua.test_all(CudaRBMs; ambiguities=false)
end
