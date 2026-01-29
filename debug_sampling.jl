using Pkg; Pkg.activate(".")
using GalaxyGenerator.MassFunctions: RedshiftMassFunction, MassFunctionSampler, integrate
using Cosmology: Planck18
using Random: Random

# Create test mass function that decreases with redshift
struct TestMF{T} <: RedshiftMassFunction{T}
    ϕ::T
    α::T
    Mstar0::T
end
function (s::TestMF)(Mstar, z)
    mass_ratio = Mstar / s.Mstar0
    # Mass function decreases with redshift: (1+z)^(-1)
    return log(10) * s.ϕ * (1 + z)^(-1) * (mass_ratio)^(s.α + 1) * exp(-mass_ratio)
end

ϕ, α, Mstar0 = 0.003, -1.3, 1e11
mf = TestMF(ϕ, α, Mstar0)

# Check actual integrated counts
Mmin, Mmax = 1e10, 1e12
count_low_z = integrate(mf, Planck18, Mmin, Mmax, 0.0, 1.0)
count_high_z = integrate(mf, Planck18, Mmin, Mmax, 1.0, 2.0)

println("Integrated galaxy counts:")
println("  z ∈ [0.0, 1.0]: ", count_low_z)
println("  z ∈ [1.0, 2.0]: ", count_high_z)
println("  Ratio (low/high): ", count_low_z / count_high_z)

# The cosmological volume increases with redshift, so even though the mass function
# decreases, the total number of galaxies might still be higher at high z
println("\nNote: Comoving volume increases with redshift,")
println("so total galaxy counts might not follow mass function trend directly.")
