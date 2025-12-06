"""Module containing modules for cosmological stellar mass functions."""
module MassFunctions

using ..GalaxyGenerator: inverse_cdf!

using ArgCheck: @argcheck, @check
using Cosmology: AbstractCosmology, comoving_volume, comoving_volume_element
using Compat: logrange
using Random: Random, default_rng, AbstractRNG
using IrrationalConstants: logten
using QuadGK: quadgk, alloc_segbuf
using Unitful: ustrip # , Quantity, @u_str
using UnitfulAstro: Mpc

# """Buffer for QuadGK integration over stellar mass."""
# const mbuf = alloc_segbuf(size=5)
# """Buffer for QuadGK integration over redshift."""
# const zbuf = alloc_segbuf(size=5)

export integrate, SchechterMassFunction, DoubleSchechterMassFunction

"""Abstract type for cosmological stellar mass functions (e.g., [`SchechterMassFunction`](@ref)).

# Units
Cosmological stellar mass functions are typically defined in logarithmic units of number density per dex in stellar mass per volume (e.g., Mpc⁻³ dex⁻¹). Unless otherwise stated, all mass functions in this module follow this convention.

# Required Methods
All subtypes should implement methods like `(s::SchechterMassFunction)(Mstar)` to evaluate the mass function at a given stellar mass, and support random sampling via `rand(s::AbstractMassFunction)`. A generic implementation of `rand` is provided here that assumes the existence of an inverse CDF interpolator stored in the `icdf` field of the subtype (so `model.icdf(x)` will return the inverse CDF of the mass function at `x`)."""
abstract type AbstractMassFunction{T} end
# Generic rand implementation assumes the existence of rand(rng, s::AbstractMassFunction)
function Random.rand(rng::Random.AbstractRNG, s::AbstractMassFunction, dims::Dims)
    return reshape([rand(rng, s) for _ in 1:prod(dims)], dims)
end

"""
`ConstantMassFunction` is an abstract type for mass functions that are constant with redshift.
"""
abstract type ConstantMassFunction{T} <: AbstractMassFunction{T} end
function Random.rand(rng::Random.AbstractRNG, s::ConstantMassFunction)
    icdf = s.icdf
    if isnothing(icdf)
        error("Provided mass function has no inverse CDF buffer; please create new instance with `mmin` and `mmax` arguments.")
    else
        icdf(rand(rng))
    end
end

"""
`RedshiftMassFunction` is an abstract type for mass functions that have dependence on both stellar mass and redshift. Instances should be callable as `model(Mstar, z)` and return the value of the mass function at stellar mass `Mstar` and redshift `z`.
"""
abstract type RedshiftMassFunction{T} <: AbstractMassFunction{T} end
function Random.rand(rng::Random.AbstractRNG, s::RedshiftMassFunction)
    icdf = s.icdf
    if isnothing(icdf)
        error("Provided mass function has no inverse CDF buffer; please create new instance with `mmin` and `mmax` arguments.")
    else
        icdf(rand(rng), rand(rng))
    end
end


"""
    integrate(model::ConstantMassFunction, mmin, mmax; npoints::Int=100)
Numerically integrates the provided stellar mass function `model` between stellar masses `mmin` and `mmax` using `npoints` to sample the mass function. Assumes the mass function is defined in logarithmic units (number density per dex in stellar mass per volume). Returns the number density per volume of galaxies with stellar masses in this range.

```jldoctest
julia> using GalaxyGenerator: SchechterMassFunction, integrate

julia> s = SchechterMassFunction(0.003, -1.3, 1e11);

julia> n_density = integrate(s, 1e9, 1e12; npoints=1000);  # Integrate between 10^9 and 10^12 Msun

julia> isapprox(n_density, 0.027; rtol=1e-3)
true
```
"""
function integrate(model::ConstantMassFunction, mmin, mmax; npoints::Int=100)
    @argcheck mmin < mmax "mmin must be less than mmax"
    # masses = logrange(mmin, mmax, npoints)
    # # In-place trapezoidal integration
    # integral = zero(first(masses))
    # prev_mass = first(masses)
    # prev_value = model(prev_mass)
    # for i in eachindex(masses)[begin+1:end]
    #     curr_mass = masses[i]
    #     curr_value = model(curr_mass)
    #     logdx = log10(curr_mass) - log10(prev_mass)
    #     avg_value = (curr_value + prev_value) / 2
    #     integral += avg_value * logdx
    #     prev_mass, prev_value = curr_mass, curr_value
    # end
    # return integral
    return quadgk(x -> model(exp10(x)), log10(mmin), log10(mmax))[1]
end
"""
    integrate(model::RedshiftMassFunction, mmin, mmax, z; npoints::Int=100)
For a redshift-dependent `model::RedshiftMassFunction`, numerically integrates the stellar mass function between stellar masses `mmin` and `mmax` at redshift `z` using `npoints` to sample the mass function. Assumes the mass function is defined in logarithmic units (number density per dex in stellar mass per volume). Returns the number density per volume of galaxies with stellar masses in this range at redshift `z`.
"""
function integrate(model::RedshiftMassFunction, mmin, mmax, z; npoints::Int=100)
    @argcheck mmin < mmax "mmin must be less than mmax"
    # masses = logrange(mmin, mmax, npoints)
    # # In-place trapezoidal integration
    # integral = zero(first(masses))
    # prev_mass = first(masses)
    # prev_value = model(prev_mass, z)
    # for i in eachindex(masses)[begin+1:end]
    #     curr_mass = masses[i]
    #     curr_value = model(curr_mass, z)
    #     logdx = log10(curr_mass) - log10(prev_mass)
    #     avg_value = (curr_value + prev_value) / 2
    #     integral += avg_value * logdx
    #     prev_mass, prev_value = curr_mass, curr_value
    # end
    # return integral
    return quadgk(x -> model(exp10(x), z), log10(mmin), log10(mmax))[1]
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
function integrate(model::ConstantMassFunction, cosmo::AbstractCosmology, mmin, mmax, z; npoints::Int=100)
    @argcheck mmin < mmax "mmin must be less than mmax"
    # Integrate the mass function over stellar masses
    mass_integral = integrate(model, mmin, mmax; npoints)
    # Convert volume to Mpc³ without units
    volume = ustrip(Mpc^3, comoving_volume(cosmo, z))
    return mass_integral * volume
end
function integrate(model::ConstantMassFunction, cosmo::AbstractCosmology, mmin, mmax, z1, z2; npoints::Int=100)
    @argcheck mmin < mmax "mmin must be less than mmax"
    @argcheck z1 < z2 "z1 must be less than z2"
    # Integrate the mass function over stellar masses
    mass_integral = integrate(model, mmin, mmax; npoints)
    # Compute the comoving volume between z1 and z2
    volume = comoving_volume(cosmo, z2) - comoving_volume(cosmo, z1)
    volume = ustrip(Mpc^3, volume) # Convert volume to Mpc³ without units
    return mass_integral * volume
end
function integrate(model::RedshiftMassFunction, cosmo::AbstractCosmology, mmin, mmax, z; npoints::Int=100)
    return integrate(model, cosmo, mmin, mmax, 1e-7, z; npoints)
end
function integrate(model::RedshiftMassFunction, cosmo::AbstractCosmology, mmin, mmax, z1, z2; npoints::Int=100)
    @argcheck mmin < mmax "mmin must be less than mmax"
    # integrate over z: N = ∫_{z1}^{z2} φcum(z) * dV/dz * dz
    # zs = logrange(z1, z2; length=npoints)
    # total = 0.0
    # for i in eachindex(zs)[begin:end-1] # Midpoint rule integration
    #     z1 = zs[i]; z2 = zs[i+1]
    #     zm = (z1 + z2) / 2
    #     dz = z2 - z1
    #     # total += φcum * dV_dz * dz
    #     N = integrate(model, mmin, mmax, zm; npoints)
    #     total += N * ustrip(Mpc^3, comoving_volume_element(cosmo, zm)) * dz
    # end
    total = quadgk(z -> begin
        N = integrate(model, mmin, mmax, z; npoints)
        N * ustrip(Mpc^3, comoving_volume_element(cosmo, z))
    end, z1, z2)[1]
    # Comoving volume element is per steradian, so multiply by 4π to get full sky
    return total * 4 * π
end


################################################
# Include specific mass function implementations
include("Schechter.jl")
export SchechterMassFunction, DoubleSchechterMassFunction

end # module