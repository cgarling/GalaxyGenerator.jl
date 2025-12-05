"""Module containing modules for cosmological stellar mass functions."""
module MassFunctions

using ..GalaxyGenerator: inverse_cdf!

using ArgCheck: @argcheck, @check
using Cosmology: AbstractCosmology, comoving_volume
using Compat: logrange
using Random: Random, default_rng, AbstractRNG
using IrrationalConstants: logten
using Unitful: ustrip # , Quantity, @u_str
using UnitfulAstro: Mpc

export integrate, SchechterMassFunction, DoubleSchechterMassFunction

"""Abstract type for cosmological stellar mass functions (e.g., [`SchechterMassFunction`](@ref)).

# Units
Cosmological stellar mass functions are typically defined in logarithmic units of number density per dex in stellar mass per volume (e.g., Mpc⁻³ dex⁻¹). Unless otherwise stated, all mass functions in this module follow this convention.

# Required Methods
All subtypes should implement methods like `(s::SchechterMassFunction)(Mstar)` to evaluate the mass function at a given stellar mass, and support random sampling via `rand(s::AbstractMassFunction)`. A generic implementation of `rand` is provided here that assumes the existence of an inverse CDF interpolator stored in the `icdf` field of the subtype (so `model.icdf(x)` will return the inverse CDF of the mass function at `x`)."""
abstract type AbstractMassFunction{T} end
function Random.rand(rng::Random.AbstractRNG, s::AbstractMassFunction)
    icdf = s.icdf
    if isnothing(icdf)
        error("Provided SchechterMassFunction has no inverse CDF buffer; please create new instance with `mmin` and `mmax` arguments.")
    else
        icdf(rand(rng))
    end
end
function Random.rand(rng::Random.AbstractRNG, s::AbstractMassFunction, dims::Dims)
    return reshape([rand(rng, s) for _ in 1:prod(dims)], dims)
end

"""
    integrate(model::AbstractMassFunction, mmin, mmax; npoints::Int=100)
Numerically integrates the provided stellar mass function `model` between stellar masses `mmin` and `mmax` using `npoints` to sample the mass function. Assumes the mass function is defined in logarithmic units (number density per dex in stellar mass per volume). Returns the number density per volume of galaxies with stellar masses in this range.

```jldoctest
julia> using GalaxyGenerator: SchechterMassFunction, integrate

julia> s = SchechterMassFunction(0.003, -1.3, 1e11);

julia> n_density = integrate(s, 1e9, 1e12; npoints=1000);  # Integrate between 10^9 and 10^12 Msun

julia> isapprox(n_density, 0.027; rtol=1e-3)
true
```
"""
function integrate(model::AbstractMassFunction, mmin, mmax; npoints::Int=100)
    @argcheck mmin < mmax "mmin must be less than mmax"
    masses = logrange(mmin, mmax, npoints)
    values = model.(masses)
    integral = zero(first(masses))
    for i in eachindex(masses)[begin+1:end]
        logdx = log10(masses[i]) - log10(masses[i-1])
        avg_value = (values[i] + values[i-1]) / 2
        integral += avg_value * logdx
    end
    return integral
end

"""
    integrate(model::AbstractMassFunction, cosmo::AbstractCosmology, mmin, mmax, z1 [, z2]; npoints::Int=100)
Numerically integrates the provided stellar mass function `model` between stellar masses `mmin` and `mmax`. If one redshift `z1` is provided, then the integration is performed from `z=0` to `z=z1`. If two redshifts `z1` and `z2` are provided, then the integration is performed *between* those two redshifts.

# Arguments
- `model`: The mass function to integrate.
- `cosmo`: The cosmology model to use for comoving volume calculations.
- `mmin`, `mmax`: Stellar mass limits.
- `z1`, `z2`: Redshift limits.

# Keyword Arguments
- `npoints`: Number of points for numerical integration over stellar mass.

# Returns
The expectation value for the number of galaxies in the specified mass and redshift ranges.

# Examples
```jldoctest integrate
julia> using GalaxyGenerator: SchechterMassFunction, integrate

julia> using Cosmology: Planck18, comoving_volume

julia> s = SchechterMassFunction(0.003, -1.3, 1e11);

julia> N = integrate(s, Planck18, 1e9, 1e12, 1.0; npoints=100);

julia> isapprox(N, 4.428e9; rtol=1e-3)
true
```
We can also integrate between two redshifts; the following calculates the same quantity as above but explicitly between `z=0` and `z=1`, though it is slightly slower as it calculates the difference between two comoving volumes at each redshift:

```jldoctest integrate
julia> isapprox(N, integrate(s, Planck18, 1e9, 1e12, 0.0, 1.0; npoints=100))
true
```
"""
function integrate(model::AbstractMassFunction, cosmo::AbstractCosmology, mmin, mmax, z; npoints::Int=100)
    @argcheck mmin < mmax "mmin must be less than mmax"
    # Integrate the mass function over stellar masses
    mass_integral = integrate(model, mmin, mmax; npoints)
    # Convert volume to Mpc³ without units
    volume = ustrip(Mpc^3, comoving_volume(cosmo, z))
    # Return the total number of galaxies
    return mass_integral * volume
end
function integrate(model::AbstractMassFunction, cosmo::AbstractCosmology, mmin, mmax, z1, z2; npoints::Int=100)
    @argcheck mmin < mmax "mmin must be less than mmax"
    @argcheck z1 < z2 "z1 must be less than z2"
    
    # Integrate the mass function over stellar masses
    mass_integral = integrate(model, mmin, mmax; npoints)
    
    # Compute the comoving volume between z1 and z2
    volume = comoving_volume(cosmo, z2) - comoving_volume(cosmo, z1)
    volume = ustrip(Mpc^3, volume) # Convert volume to Mpc³ without units
    
    # Return the total number of galaxies
    return mass_integral * volume
end


################################################
# Include specific mass function implementations
include("Schechter.jl")
export SchechterMassFunction, DoubleSchechterMassFunction

end # module