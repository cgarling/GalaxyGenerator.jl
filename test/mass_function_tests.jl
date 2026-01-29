using Cosmology: Planck18, comoving_volume_element
using Compat: logrange
using GalaxyGenerator.MassFunctions: RedshiftMassFunction, SchechterMassFunction, integrate, MassFunctionSampler
import Random
using QuadGK: quadgk
import StatsBase
using Test
using Unitful: ustrip
using UnitfulAstro: Mpc

const cosmo = Planck18

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

@testset "integrals" begin
    @testset "RedshiftMassFunction" begin
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

@testset "sampling" begin
    @testset "ConstantMassFunction sampling" begin
        """Test that sampling from a mass function produces the expected distribution."""
        # Create a Schechter mass function
        ϕ, α, Mstar0 = 0.003, -1.3, 1e11
        s = SchechterMassFunction(ϕ, α, Mstar0)
        
        Mmin, Mmax = 1e9, 1e12
        sampler = MassFunctionSampler(s, Mmin, Mmax; npoints=1000)
        
        # Sample a large number of masses
        Random.seed!(42)
        n_samples = 100000
        samples = rand(sampler, n_samples)

        # Check that all samples are within bounds
        @test all(Mmin .<= samples .<= Mmax)
        
        # Bin the samples and compare to expected distribution
        n_bins = 20
        bin_edges = logrange(Mmin, Mmax, n_bins+1)
        bin_centers = sqrt.(bin_edges[1:end-1] .* bin_edges[2:end])
        
        # Make histogram of samples in bins
        hist = StatsBase.fit(StatsBase.Histogram, samples, bin_edges)
        bin_counts = hist.weights
        
        # Normalize to get empirical probability density per dex
        bin_widths_dex = diff(log10.(bin_edges))
        empirical_density = bin_counts ./ (n_samples .* bin_widths_dex)
        
        # Expected probability density (normalized mass function)
        total_integral = integrate(s, Mmin, Mmax)
        expected_density = [s(m) / total_integral for m in bin_centers]
        
        # Test that empirical and expected densities match within statistical error
        # Use a relative tolerance of 10% for each bin (conservative given sampling variance)
        for i in 1:n_bins
            if expected_density[i] > 0.01 * maximum(expected_density)  # Only check bins with significant density
                @test empirical_density[i] ≈ expected_density[i] rtol=0.05
            end
        end
    end
    
    @testset "RedshiftMassFunction sampling" begin
        ϕ, α, Mstar0, α0 = 0.003, -1.3, 1e11, 1e-3
        mf = TestSchechter(ϕ, α, Mstar0, α0)
        
        Mmin, Mmax = 1e10, 1e12
        zmin, zmax = 0.0, 2.0
        npoints_mass = 50
        npoints_redshift = 50
        sampler = MassFunctionSampler(mf, cosmo, Mmin, Mmax, zmin, zmax; 
                                      npoints_mass=npoints_mass,
                                      npoints_redshift=npoints_redshift)
        
        # Sample multiple objects
        Random.seed!(42)
        n_samples = 100000
        samples = rand(sampler, n_samples)
        
        # Check that all samples are within bounds
        @test all(Mmin .<= samples[1, :] .<= Mmax)
        @test all(zmin .<= samples[2, :] .<= zmax)
        
        # Check that cumulative distribution of samples matches expected CDF
        # Count samples in each mass, redshift bin
        mass_bin_edges = sampler.mass_grid[:, 1]
        redshift_bin_edges = sampler.redshift_grid
        hist = StatsBase.fit(StatsBase.Histogram, (samples[1, :], samples[2, :]), (mass_bin_edges, redshift_bin_edges))
        bin_counts = hist.weights

        # Calculate expected counts in each bin via integration
        expected_counts = zeros(size(bin_counts))
        for i in eachindex(mass_bin_edges)[1:end-1]
            for j in eachindex(redshift_bin_edges)[1:end-1]
                m1, m2 = mass_bin_edges[i], mass_bin_edges[i+1]
                z1, z2 = redshift_bin_edges[j], redshift_bin_edges[j+1]
                expected_counts[i, j] = integrate(mf, cosmo, m1, m2, z1, z2)
            end
        end

        # Renormalize expected counts to total number of samples
        expected_counts .*= n_samples / sum(expected_counts)

        # Make residual significance, difference divided by sqrt(expected)
        signif = (bin_counts .- expected_counts) ./ sqrt.(expected_counts .+ 1e-10)
        
        # If difference between expectations and samples is just Poisson noise,
        # then the variance of signif should be ~1 and mean ~0
        @test abs(StatsBase.mean(signif)) < 0.05
        @test abs(StatsBase.var(signif) - 1.0) < 0.05
    end
end