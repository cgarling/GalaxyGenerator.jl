using GalaxyGenerator
using Test
using SafeTestsets: @safetestset

@testset "GalaxyGenerator.jl" begin
    @safetestset "doctests" include("doctests.jl")
    @safetestset "IGM" include("igm_tests.jl")
    @safetestset "MassFunctions" include("mass_function_tests.jl")
end
