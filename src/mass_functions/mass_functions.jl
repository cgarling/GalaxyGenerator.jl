"""Module containing modules for cosmological stellar mass functions."""
module MassFunctions

using ..GalaxyGenerator: interp_lin, interp_log

using ArgCheck: @argcheck, @check
using Cosmology: AbstractCosmology, comoving_volume, comoving_volume_element
using Compat: logrange
using HCubature: hcubature
using Random: Random, default_rng, AbstractRNG
using IrrationalConstants: logten
using QuadGK: quadgk
using Unitful: ustrip
using UnitfulAstro: Mpc

export integrate, SchechterMassFunction, DoubleSchechterMassFunction, MassFunctionSampler

"""Abstract type for cosmological stellar mass functions (e.g., [`SchechterMassFunction`](@ref)).

# Units
Cosmological stellar mass functions are typically defined in logarithmic units of number density per dex in stellar mass per volume (e.g., Mpc⁻³ dex⁻¹). Unless otherwise stated, all mass functions in this module follow this convention.

# Required Methods
All subtypes should implement methods like `(s::SchechterMassFunction)(Mstar)` to evaluate the mass function at a given stellar mass, and support random sampling via `rand(s::AbstractMassFunction)`."""
abstract type AbstractMassFunction{T} end

# Generic rand implementation assumes the existence of rand(rng, s::AbstractMassFunction)
function Random.rand(rng::Random.AbstractRNG, s::AbstractMassFunction, dims::Dims)
    return reshape([rand(rng, s) for _ in 1:prod(dims)], dims)
end

###################################
# Redshift-constant mass functions
"""
`ConstantMassFunction` is an abstract type for mass functions that are constant with redshift. A generic implementation of `rand` is defined that assumes the existence of an inverse CDF interpolator stored in the `icdf` field of the subtype (so `model.icdf(x)` will return the inverse CDF of the mass function at `x`).
"""
abstract type ConstantMassFunction{T} <: AbstractMassFunction{T} end

Random.rand(rng::Random.AbstractRNG, s::ConstantMassFunction) = error("Random sampling of `ConstantMassFunction` subtypes requires instantiating a `MassFunctionSampler` from your stellar mass model.")

struct ConstantMassFunctionSampler{T, A <: AbstractVector{T}, B <: AbstractVector, C <: ConstantMassFunction} <: ConstantMassFunction{T}
    x::A  # CDF values
    y::B  # stellar masses corresponding to CDF values
    mf::C # The mass function model
end

"""
    MassFunctionSampler(model::ConstantMassFunction, mmin, mmax; npoints=1000) <: ConstantMassFunction

Creates a sampler for any `ConstantMassFunction` via inverse CDF sampling. 
Implements the `Random.rand` interface for sampling from the mass function.
For speed a look-up table of the inverse CDF is used for sampling; coverage 
of this look-up table is defined by the keyword arguments `mmin` and `mmax` 
which give the limits of the look-up table in solar masses. The number of 
elements in this look-up table is `npoints`; more points gives greater 
sampling accuracy at the cost of increased memory and decreased sampling speed. 
The default `npoints=1000` is typically sufficient.

# Arguments
- `model`: The `ConstantMassFunction` to sample from.
- `mmin`, `mmax`: The range of stellar masses (in solar masses) for the inverse CDF.
- `npoints`: Number of points to use for the inverse CDF calculation (default: 1000).

# Returns
A `MassFunctionSampler` struct containing the inverse CDF `x` and `y` values.

```jldoctest
julia> using GalaxyGenerator: SchechterMassFunction, MassFunctionSampler

julia> s = SchechterMassFunction(0.003, -1.3, 1e11);

julia> sampler = MassFunctionSampler(s, 1e9, 1e12);

julia> rand(sampler) isa Float64 # Sample one value
true

julia> rand(sampler, 5) isa Vector{Float64} # Sampling multiple values
true

julia> s(1e9) == sampler(1e9) # sampler is also a mass function
true
```
"""
function MassFunctionSampler(model::ConstantMassFunction, mmin, mmax; npoints::Int=1000)
    x = logrange(mmin, mmax, npoints)
    cdf = model.(x)
    cumsum!(@view(cdf[2:end]), @views (cdf[2:end] .+ cdf[1:end-1]) ./ 2 .* diff(x))
    cdf[1] = 0
    cdf ./= last(cdf)
    T = typeof(first(cdf))
    return ConstantMassFunctionSampler{T, typeof(x), typeof(cdf), typeof(model)}(x, cdf, model)
end

(model::ConstantMassFunctionSampler)(Mstar) = model.mf(Mstar)

function Random.rand(rng::AbstractRNG, sampler::ConstantMassFunctionSampler)
    u = rand(rng)  # Uniform random number in [0, 1]
    return interp_lin(sampler.y, sampler.x, u; extrapolate=false)
end

###################################
# Redshift-dependent mass functions

"""
`RedshiftMassFunction` is an abstract type for mass functions that have dependence on both stellar mass and redshift. Instances should be callable as `model(Mstar, z)` and return the value of the mass function at stellar mass `Mstar` and redshift `z`.
"""
abstract type RedshiftMassFunction{T} <: AbstractMassFunction{T} end

# Random.rand(rng::Random.AbstractRNG, s::RedshiftMassFunction) = error("Random sampling of `RedshiftMassFunction` subtypes requires instantiating a `MassFunctionSampler` from your stellar mass model.")


######################
# Integration routines
"""
    integrate(model::ConstantMassFunction, mmin, mmax; kws...)
Numerically integrates the provided stellar mass function `model` between stellar masses `mmin` and `mmax`. Assumes the mass function is defined in logarithmic units (number density per dex in stellar mass per volume). Returns the number density per volume of galaxies with stellar masses in this range. Keyword arguments `kws...` are passed to `QuadGK.quadgk` to perform the numerical integration.
"""
function integrate(model::ConstantMassFunction, mmin, mmax; kws...)
    @argcheck mmin < mmax "mmin must be less than mmax"
    return quadgk(x -> model(exp10(x)), log10(mmin), log10(mmax); kws...)[1]
end

"""
    integrate(model::RedshiftMassFunction, mmin, mmax, z; kws...)
For a redshift-dependent `model::RedshiftMassFunction`, numerically integrates the stellar mass function between stellar masses `mmin` and `mmax` at redshift `z`. Assumes the mass function is defined in logarithmic units (number density per dex in stellar mass per volume). Returns the number density per volume of galaxies with stellar masses in this range at redshift `z`. Keyword arguments `kws...` are passed to `QuadGK.quadgk` to perform the numerical integration.
"""
function integrate(model::RedshiftMassFunction, mmin, mmax, z; kws...)
    @argcheck mmin < mmax "mmin must be less than mmax"
    return quadgk(x -> model(exp10(x), z), log10(mmin), log10(mmax); kws...)[1]
end

"""
    integrate(model::AbstractMassFunction, cosmo::AbstractCosmology, mmin, mmax, z1 [, z2]; kws...)
Numerically integrates the provided stellar mass function `model` between stellar masses `mmin` and `mmax`. If one redshift `z1` is provided, then the integration is performed from `z=0` to `z=z1`. If two redshifts `z1` and `z2` are provided, then the integration is performed *between* those two redshifts.

# Arguments
- `model`: The mass function to integrate.
- `cosmo`: The cosmology model to use for comoving volume calculations.
- `mmin`, `mmax`: Stellar mass limits.
- `z1`, `z2`: Redshift limits.

# Keyword Arguments
- `kws...`: Keyword arguments `kws...` are passed to `QuadGK.quadgk` to perform the numerical integration.

# Returns
The expectation value for the number of galaxies in the specified mass and redshift ranges.

# Examples
```jldoctest integrate
julia> using GalaxyGenerator: SchechterMassFunction, integrate

julia> using Cosmology: Planck18, comoving_volume

julia> s = SchechterMassFunction(0.003, -1.3, 1e11);

julia> N = integrate(s, Planck18, 1e9, 1e12, 1.0);

julia> isapprox(N, 4.428e9; rtol=1e-3)
true
```
We can also integrate between two redshifts; the following calculates the same quantity as above but explicitly between `z=0` and `z=1`, though it is slightly slower as it calculates the difference between two comoving volumes at each redshift:

```jldoctest integrate
julia> isapprox(N, integrate(s, Planck18, 1e9, 1e12, 0.0, 1.0))
true
```
"""
function integrate(model::ConstantMassFunction, cosmo::AbstractCosmology, mmin, mmax, z; kws...)
    @argcheck mmin < mmax "mmin must be less than mmax"
    # Integrate the mass function over stellar masses
    mass_integral = integrate(model, mmin, mmax; kws...)
    # Convert volume to Mpc³ without units
    volume = ustrip(Mpc^3, comoving_volume(cosmo, z))
    return mass_integral * volume
end
function integrate(model::ConstantMassFunction, cosmo::AbstractCosmology, mmin, mmax, z1, z2; kws...)
    @argcheck mmin < mmax "mmin must be less than mmax"
    @argcheck z1 < z2 "z1 must be less than z2"
    # Integrate the mass function over stellar masses
    mass_integral = integrate(model, mmin, mmax; kws...)
    # Compute the comoving volume between z1 and z2
    volume = comoving_volume(cosmo, z2) - comoving_volume(cosmo, z1)
    volume = ustrip(Mpc^3, volume) # Convert volume to Mpc³ without units
    return mass_integral * volume
end

function integrate(model::RedshiftMassFunction, cosmo::AbstractCosmology, mmin, mmax, z; kws...)
    return integrate(model, cosmo, mmin, mmax, zero(z), z; kws...)
end
function integrate(model::RedshiftMassFunction, cosmo::AbstractCosmology, mmin, mmax, z1, z2; kws...)
    @argcheck mmin < mmax "mmin must be less than mmax"
    # integrate over z: N = ∫_{z1}^{z2} φcum(z) * dV/dz * dz
    integrand(x) = begin
        logMstar, z = x
        N = model(exp10(logMstar), z)
        N * ustrip(Mpc^3, comoving_volume_element(cosmo, z))
    end
    total = hcubature(integrand, (log10(mmin), z1), (log10(mmax), z2); kws...)[1]
    # Comoving volume element is per steradian, so multiply by 4π to get full sky
   return total * 4 * π
end


################################################
# Include specific mass function implementations
include("Schechter.jl")
export SchechterMassFunction, DoubleSchechterMassFunction
include("BinnedRedshiftMassFunction.jl")

end # module