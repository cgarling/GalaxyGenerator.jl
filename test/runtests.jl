using GalaxyGenerator
using Test
using SafeTestsets: @safetestset

@testset "GalaxyGenerator.jl" begin
    @safetestset "EGG Module" include("egg_tests.jl")
end
