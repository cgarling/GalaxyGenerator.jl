using GalaxyGenerator
using Test
using SafeTestsets: @safetestset

@testset "GalaxyGenerator.jl" begin
    @safetestset "doctests" include("doctests.jl")
    @safetestset "IGM" include("igm_tests.jl")
end
