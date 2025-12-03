import GalaxyGenerator as GG
import Unitful as u
using Test

@testset "IGM" begin
    x = 0.35:0.001:0.55 # microns
    x_u = x * u.μm
    z_arr = 0.0:0.1:5.5
    @testset "NoIGM" begin
        s = GG.EGG.NoIGM()
        for z in z_arr
            @test all(GG.EGG.transmission.(s, z, x) .== 1)
            @test all(GG.EGG.transmission.(s, z, x_u) .== 1)
            @test all(GG.EGG.tau.(s, z, x) .== 0)
            @test all(GG.EGG.tau.(s, z, x_u) .== 0)
        end
    end
    @testset "Madau1995IGM" begin
        s = GG.EGG.Madau1995IGM()
        @test isapprox(GG.EGG.transmission(s, 3.5, 0.11), 0.617783510515702; rtol=1e-5)
        @test isapprox(GG.EGG.tau(s, 3.5, 0.11), -log(0.617783510515702); rtol=1e-5)
        for z in z_arr
            r = GG.EGG.transmission.(s, z, x)
            @test r isa Vector{Float64}
            @test maximum(r) <= 1.0
            @test minimum(r) >= 0.0
            # Test Unitful input
            @test GG.EGG.transmission.(s, z, x_u) == r
        end
    end
end