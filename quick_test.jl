# Quick test of the fixes
include("src/GalaxyGenerator.jl")
using .GalaxyGenerator.MassFunctions: SchechterMassFunction, MassFunctionSampler
using Random: Random

println("Creating Schechter mass function...")
ϕ, α, Mstar0 = 0.003, -1.3, 1e11
s = SchechterMassFunction(ϕ, α, Mstar0)

println("Creating sampler...")
Mmin, Mmax = 1e9, 1e12
sampler = MassFunctionSampler(s, Mmin, Mmax; npoints=100)

println("Checking CDF properties...")
cdf_values = sampler.y

# Check monotonicity
is_monotonic = all(diff(cdf_values) .>= 0)
println("  Is monotonic: ", is_monotonic)
if !is_monotonic
    println("  ERROR: CDF is not monotonic!")
    exit(1)
end

# Check normalization
starts_at_zero = abs(cdf_values[1]) < 1e-10
ends_at_one = abs(cdf_values[end] - 1.0) < 1e-10
println("  Starts at 0: ", starts_at_zero, " (value: ", cdf_values[1], ")")
println("  Ends at 1: ", ends_at_one, " (value: ", cdf_values[end], ")")

if !starts_at_zero || !ends_at_one
    println("  ERROR: CDF normalization is incorrect!")
    exit(1)
end

println("\nTesting sampling...")
Random.seed!(42)
samples = [rand(sampler) for _ in 1:1000]

# Check all samples are in range
in_range = all(Mmin .<= samples .<= Mmax)
println("  All samples in range: ", in_range)
if !in_range
    println("  ERROR: Some samples out of range!")
    exit(1)
end

println("\n✅ All basic tests passed!")
