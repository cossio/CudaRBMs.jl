import Aqua
import CudaRBMs

using Test: @testset

@testset "aqua" begin
    Aqua.test_all(
        CudaRBMs;
        ambiguities=(exclude=[reshape, get!, trunc, findall, Base.unsafe_convert],),
        stale_deps=(ignore=[:cuDNN],),
    )
end
