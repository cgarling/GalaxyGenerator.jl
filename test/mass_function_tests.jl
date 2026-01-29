using Cosmology: Planck18, comoving_volume_element
using GalaxyGenerator.MassFunctions: RedshiftMassFunction, SchechterMassFunction, integrate, MassFunctionSampler
using QuadGK: quadgk
using Random: Random
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
        samples = [rand(sampler) for _ in 1:n_samples]
        
        # Bin the samples and compare to expected distribution
        n_bins = 20
        bin_edges = exp10.(range(log10(Mmin), log10(Mmax), length=n_bins+1))
        bin_centers = sqrt.(bin_edges[1:end-1] .* bin_edges[2:end])
        
        # Count samples in each bin
        bin_counts = zeros(n_bins)
        for sample in samples
            for i in 1:n_bins
                if i == n_bins
                    # Last bin includes upper boundary
                    if bin_edges[i] <= sample <= bin_edges[i+1]
                        bin_counts[i] += 1
                        break
                    end
                else
                    if bin_edges[i] <= sample < bin_edges[i+1]
                        bin_counts[i] += 1
                        break
                    end
                end
            end
        end
        
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
                @test empirical_density[i] ≈ expected_density[i] rtol=0.1
            end
        end
    end
    
    @testset "CDF monotonicity" begin
        """Test that the CDF is monotonically increasing."""
        ϕ, α, Mstar0 = 0.003, -1.3, 1e11
        s = SchechterMassFunction(ϕ, α, Mstar0)
        
        Mmin, Mmax = 1e9, 1e12
        sampler = MassFunctionSampler(s, Mmin, Mmax; npoints=100)
        
        # Check that the CDF is monotonically increasing
        cdf_values = sampler.y
        @test all(diff(cdf_values) .>= 0)
        @test cdf_values[1] ≈ 0.0 atol=1e-10
        @test cdf_values[end] ≈ 1.0 rtol=1e-10
    end
    
    @testset "RedshiftMassFunction sampling" begin
        """Test sampling from a redshift-dependent mass function."""
        # Create a simple redshift-dependent mass function
        struct SimpleRedshiftMF{T} <: RedshiftMassFunction{T}
            ϕ::T
            α::T
            Mstar0::T
        end
        function (s::SimpleRedshiftMF)(Mstar, z)
            mass_ratio = Mstar / s.Mstar0
            # Mass function decreases with redshift
            return log(10) * s.ϕ * (1 + z)^(-1) * (mass_ratio)^(s.α + 1) * exp(-mass_ratio)
        end
        
        ϕ, α, Mstar0 = 0.003, -1.3, 1e11
        mf = SimpleRedshiftMF(ϕ, α, Mstar0)
        
        Mmin, Mmax = 1e10, 1e12
        zmin, zmax = 0.0, 2.0
        sampler = MassFunctionSampler(mf, cosmo, Mmin, Mmax, zmin, zmax; 
                                      npoints_mass=50, npoints_redshift=50)
        
        # Sample multiple objects
        Random.seed!(42)
        n_samples = 10000
        samples = rand(sampler, n_samples)
        
        # Check that all samples are within bounds
        @test all(Mmin .<= samples[1, :] .<= Mmax)
        @test all(zmin .<= samples[2, :] .<= zmax)
        
        # Check that redshift distribution roughly matches expectation
        # Even though the mass function decreases with redshift (∝ (1+z)^(-1)),
        # the comoving volume increases faster, so there are more galaxies at higher z
        n_low_z = sum(samples[2, :] .< 1.0)
        n_high_z = sum(samples[2, :] .>= 1.0)
        @test n_high_z > n_low_z  # More galaxies at higher z due to volume effects
    end
end