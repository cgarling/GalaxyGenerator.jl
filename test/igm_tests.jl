using GalaxyGenerator.IGM
import Unitful as u
using Test

@testset "IGM" begin
    x = 0.35:0.001:0.55 # microns
    x_u = x * u.μm
    z_arr = 0.0:0.1:5.5
    @testset "NoIGM" begin
        s = NoIGM()
        for z in z_arr
            @test all(transmission.(s, z, x) .== 1)
            @test all(transmission.(s, z, x_u) .== 1)
            @test all(tau.(s, z, x) .== 0)
            @test all(tau.(s, z, x_u) .== 0)
        end
    end
    @testset "Madau1995IGM" begin
        s = Madau1995IGM()
        @test isapprox(transmission(s, 3.5, 0.11), 0.617783510515702; rtol=1e-5)
        @test isapprox(tau(s, 3.5, 0.11), -log(0.617783510515702); rtol=1e-5)
        for z in z_arr
            r = transmission.(s, z, x)
            @test r isa Vector{Float64}
            @test maximum(r) <= 1.0
            @test minimum(r) >= 0.0
            # Test Unitful input
            @test transmission.(s, z, x_u) == r
        end
    end
    @testset "Inoue2014IGM" begin
        s = Inoue2014IGM()
        @test isapprox(transmission(s, 3.5, 0.11), 0.6516501946923478; rtol=1e-5)
        @test isapprox(tau(s, 3.5, 0.11), -log(0.6516501946923478); rtol=1e-5)
        for z in z_arr
            r = transmission.(s, z, x)
            @test r isa Vector{Float64}
            @test maximum(r) <= 1.0
            @test minimum(r) >= 0.0
            # Test Unitful input
            @test transmission.(s, z, x_u) == r
        end
    end
end
