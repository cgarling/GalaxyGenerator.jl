using Cosmology: Planck18, comoving_volume_element
using GalaxyGenerator.MassFunctions: RedshiftMassFunction, SchechterMassFunction, integrate
using QuadGK: quadgk
using Test
using Unitful: ustrip
using UnitfulAstro: Mpc

const cosmo = Planck18

@testset "integrals" begin
    

    @testset "RedshiftMassFunction" begin
        """Test implementation of redshift-dependent mass function integration."""
        struct TestSchechter{T} <: RedshiftMassFunction{T}
            ϕ::T
            α::T
            Mstar0::T
            α0::T
        end
        function (s::TestSchechter)(Mstar, z)
            mass_ratio = Mstar / s.Mstar0
            return log(10) * s.ϕ * (mass_ratio)^(s.α + 1) * exp(-mass_ratio) + s.α0 * z
        end
        
        ϕ, α, Mstar0 = 0.003, -1.3, 1e11
        s1 = SchechterMassFunction(ϕ, α, Mstar0)
        α0 = 1e-3
        s2 = TestSchechter(ϕ, α, Mstar0, α0)
        Mmin, Mmax = 1e9, 1e12
        # At z=0, these should give the same answer
        # When integrating over log(Mstar)
        r1 = integrate(s1, Mmin, Mmax)
        r2 = integrate(s2, Mmin, Mmax, 0.0)
        @test r1 ≈ r2 rtol=1e-7
        # At redshift z2, they will differ by the integral
        # ∫_{logMmin}^{logMmax} α0 * z2 * d(logMstar)
        z2 = 1.0
        v_int = quadgk(logMstar -> α0 * z2, log10(Mmin), log10(Mmax))[1]
        @test r1 ≈ integrate(s2, Mmin, Mmax, z2) - v_int atol=1e-7

        # Test integration over stellar mass *and* redshift, 
        # first with TestSchechter with no redshift evolution α0=0,
        # which should match the constant SchechterMassFunction
        s3 = TestSchechter(ϕ, α, Mstar0, 0.0)
        @test integrate(s3, cosmo, Mmin, Mmax, 0.0, 1.0) ≈ integrate(s1, cosmo, Mmin, Mmax, 0.0, 1.0) rtol=1e-7
    end

end