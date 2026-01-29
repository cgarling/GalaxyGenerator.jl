"""Module containing modules for cosmological stellar mass functions."""
module MassFunctions

using ..GalaxyGenerator: interp_lin, interp_log, find_bin

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
    x::A  # Stellar mass grid
    y::B  # CDF values corresponding to stellar masses
    model::C # The mass function model
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
A `ConstantMassFunctionSampler` struct containing the inverse CDF `x` and `y` values.

# Notes
As is common for numerical inverse transform sampling, the first few bins of stellar mass (where the CDF is very low) will likely be slightly undersampled (i.e., fewer samples than expected).

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
    f = model.(x)
    # Compute trapezoidal integration: ∫ f(log10(M)) d(log10(M))
    # First compute the trapezoid areas, then cumsum
    trap_areas = @views (f[2:end] .+ f[1:end-1]) ./ 2 .* diff(log10.(x))
    cdf = similar(f)
    cdf[1] = 0
    cumsum!(@view(cdf[2:end]), trap_areas)
    cdf ./= last(cdf)
    T = typeof(first(cdf))
    return ConstantMassFunctionSampler{T, typeof(x), typeof(cdf), typeof(model)}(x, cdf, model)
end

(model::ConstantMassFunctionSampler)(Mstar) = model.model(Mstar)

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

struct RedshiftMassFunctionSampler{T, A <: AbstractMatrix{T}, B <: AbstractVector{T}, C <: AbstractVector{T}, D <: RedshiftMassFunction} <: RedshiftMassFunction{T}
    cdf::A  # 2D CDF grid; stellar mass CDF at every redshift
    mass_grid::A  # Stellar mass grid at every redshift
    redshift_grid::B  # Redshift grid
    z_cdf::C  # Redshift CDF
    model::D  # The mass function model
end
(model::RedshiftMassFunctionSampler)(Mstar, z) = model.model(Mstar, z)

"""
    MassFunctionSampler(model::RedshiftMassFunction, cosmo::AbstractCosmology, mmin, mmax, zmin, zmax; npoints_mass=100, npoints_redshift=100)

Creates a sampler for any `RedshiftMassFunction` by calculating the inverse CDF over the range `[mmin, mmax]` for stellar mass and `[zmin, zmax]` for redshift.
Instances implement the `Random.rand` interface for sampling from the mass function and return `(Mstar, z)`.

# Arguments
- `model`: The `RedshiftMassFunction` to sample from.
- `cosmo`: The cosmology model to use.
- `mmin`, `mmax`: The range of stellar masses (in solar masses) for the inverse CDF.
- `zmin`, `zmax`: The range of redshifts for the inverse CDF.
- `npoints_mass`: Number of points to use for the stellar mass axis (default: 100).
- `npoints_redshift`: Number of points to use for the redshift axis (default: 100).

# Returns
A `RedshiftMassFunctionSampler` instance containing the inverse CDF `x` and `y` values.

# Notes
As is common for numerical inverse transform sampling, the first few bins of stellar mass (where the CDF is very low) will likely be slightly undersampled (i.e., fewer samples than expected).

# Examples
```jldoctest mfs_redshift
julia> using GalaxyGenerator: MassFunctionSampler

julia> using GalaxyGenerator.EGG: EGGMassFunction_SF # Load EGG mass function, which is redshift-dependent

julia> using Cosmology: Planck18

julia> sampler = MassFunctionSampler(EGGMassFunction_SF, Planck18, 1e9, 1e12, 0.0, 1.0);

julia> rand(sampler) isa NTuple{2, Float64} # Sample one value, returns (Mstar, z)
true

julia> rand(sampler, 5) isa Matrix{Float64} # Sampling multiple values returned as matrix
true

julia> dims = 5
5

julia> size(rand(sampler, dims)) # Return matrix has dimensions (2, dims...)
(2, 5)

julia> sampler(1e9, 0.5) == EGGMassFunction_SF(1e9, 0.5) # sampler is also a mass function
true
```
"""
function MassFunctionSampler(model::RedshiftMassFunction, cosmo::AbstractCosmology, mmin, mmax, zmin, zmax; npoints_mass::Int=100, npoints_redshift::Int=100, kws...)
    redshift_grid = range(zmin, zmax, length=npoints_redshift)
    return MassFunctionSampler(model, cosmo, fill(mmin, npoints_redshift), fill(mmax, npoints_redshift), redshift_grid; npoints_mass, kws...)
end

"""
    MassFunctionSampler(model::RedshiftMassFunction, 
                        cosmo::AbstractCosmology, 
                        mmin::AbstractVector, 
                        mmax::AbstractVector, 
                        redshift_grid::AbstractVector; 
                        npoints_mass::Int=100, kws...)

Creates a sampler for any `RedshiftMassFunction` by calculating the inverse CDF over the range of stellar masses defined by `mmin` and `mmax` vectors at each redshift in `redshift_grid`;
i.e., the minimum and maximum stellar masses at `z = redshift_grid[i]` are `mmin[i]` and `mmax[i]` respectively. This variable stellar mass range as a function of redshift
is useful for implementing mass completeness limits that vary with redshift.

# Arguments
- `model`: The `RedshiftMassFunction` to sample from.
- `cosmo`: The cosmology model to use.
- `mmin`: The minimum stellar mass (in solar masses) at each redshift.
- `mmax`: The maximum stellar mass (in solar masses) at each redshift.
- `redshift_grid`: The redshift values to sample from.
- `npoints_mass`: The number of points to use for the stellar mass axis (default: 100).
- `kws`: Additional keyword arguments to pass to the mass function.

# Returns
A `RedshiftMassFunctionSampler` instance containing the inverse CDF `x` and `y` values.

# Examples
Assuming a constant mass-to-light ratio, the mass completeness limit of a flux-limited survey varies with redshift. Here we create a mass function sampler that accounts for this varying mass completeness limit.
We assume a the survey is complete to `Mstar = 1e9` at redshift `z=0.1` and that the mass completeness limit increases with redshift proportional to the luminosity distance with constant mass-to-light ratio.
```jldoctest mfs_redshift
julia> using GalaxyGenerator: MassFunctionSampler

julia> using GalaxyGenerator.EGG: EGGMassFunction_SF # Load EGG mass function, which is redshift-dependent

julia> using Cosmology: Planck18, luminosity_dist

julia> z = 0.1:0.01:3.0
0.1:0.01:3.0

julia> mmin = exp10(9) .* (luminosity_dist.(Planck18, z) ./ luminosity_dist(Planck18, 0.1)); # Example mass completeness limit increasing with redshift

julia> mmax = fill(1e13, length(z)); # Fixed maximum mass for sampling

julia> sampler = MassFunctionSampler(EGGMassFunction_SF, Planck18, mmin, mmax, z);

julia> rand(sampler) isa NTuple{2, Float64} # Sample one value, returns (Mstar, z)
true

julia> rand(sampler, 5) isa Matrix{Float64} # Sampling multiple values returned as matrix
true
```
"""
function MassFunctionSampler(model::RedshiftMassFunction, cosmo::AbstractCosmology, mmin::AbstractVector, mmax::AbstractVector, redshift_grid::AbstractVector; npoints_mass::Int=100, kws...)
    @argcheck issorted(redshift_grid) "`redshift_grid` must be sorted in ascending order."
    @argcheck length(mmin) == length(mmax) == length(redshift_grid) "`mmin` must have the same length as `mmax` and `redshift_grid`."
    for i in eachindex(mmin)
        @check mmin[i] < mmax[i] "`mmin` must be less than `mmax` at all redshifts, found mmin=$(mmin[i]) >= mmax=$(mmax[i]) at index $i."
    end

    npoints_redshift = length(redshift_grid)
    mass_grid = zeros(npoints_mass, npoints_redshift)
    # Fill mass grid range of masses at each redshift
    for i in axes(mass_grid, 2)
        mass_grid[:, i] .= logrange(mmin[i], mmax[i], length=npoints_mass)
    end

    # Compute differential number of galaxies in each bin of mass, redshift
    cdf = similar(mass_grid)
    # idxs = redshift_grid[1] ≈ 0 ? axes(mass_grid, 1)[begin+1:end] : axes(mass_grid, 1)
    idxs = axes(mass_grid, 1)[begin+1:end]
    Threads.@threads for i in idxs
    # for i in idxs
        for j in eachindex(redshift_grid)[begin+1:end]
            # cdf[i, j] = integrate(model, cosmo, mass_grid[i-1], mass_grid[i], redshift_grid[j-1], redshift_grid[j]; kws...)
            cdf[i, j] = integrate(model, cosmo, mass_grid[i-1,j], mass_grid[i,j], redshift_grid[j-1], redshift_grid[j]; kws...)
        end
    end
    cdf[:,1] .= 0 # Fix first column, row
    cdf[1,:] .= 0

    # To sample redshift, we need to sum over the mass axis
    # cdf[i,j] already contains integrated galaxy counts, so we just need cumsum
    z_cdf = vec(sum(cdf, dims=1))
    cumsum!(z_cdf, z_cdf)
    z_cdf ./= last(z_cdf)

    # Once we sample redshift, we need to calculate CDF of every column
    # cdf[:,i] already contains integrated galaxy counts in each mass bin,
    # so we just need cumsum (not trapezoidal integration)
    for i in axes(cdf, 2)
        col = @view cdf[:, i]
        cumsum!(col, col)
        col ./= last(col)
    end

    T = typeof(first(cdf))
    redshift_grid = convert(Vector{T}, redshift_grid)
    return RedshiftMassFunctionSampler{T, typeof(cdf), typeof(redshift_grid), typeof(z_cdf), typeof(model)}(cdf, mass_grid, redshift_grid, z_cdf, model)
end

function Random.rand(rng::AbstractRNG, sampler::RedshiftMassFunctionSampler)
    u_mass = rand(rng)  # Uniform random number for stellar mass
    u_redshift = rand(rng)  # Uniform random number for redshift

    # First interpolate redshift
    z = interp_lin(sampler.z_cdf, sampler.redshift_grid, u_redshift; extrapolate=false)

    # Given this redshift, find which redshift bin the galaxy falls into
    zidx = max(2, find_bin(z, sampler.redshift_grid))

    # Interpolation between redshift bins
    v1, v2 = @views (sampler.cdf[:, zidx], sampler.cdf[:, zidx+1])
    m1, m2 = @views (sampler.mass_grid[:, zidx], sampler.mass_grid[:, zidx+1])
    # Whenever possible, use log interpolation for mass functions
    Mstar1 = u_mass > v1[2] ? interp_log(v1, m1, u_mass; extrapolate=false) : interp_lin(v1, m1, u_mass; extrapolate=false)
    Mstar2 = u_mass > v2[2] ? interp_log(v2, m2, u_mass; extrapolate=false) : interp_lin(v2, m2, u_mass; extrapolate=false)
    Mstar = Mstar1 + (Mstar2 - Mstar1) * (z - sampler.redshift_grid[zidx]) / (sampler.redshift_grid[zidx+1] - sampler.redshift_grid[zidx])
    return Mstar, z
end

function Random.rand(rng::AbstractRNG, sampler::RedshiftMassFunctionSampler{T}, dims::Dims) where T
    # return reinterpret(reshape, T, [rand(rng, sampler) for _ in 1:prod(dims)])
    out = Array{T}(undef, 2, dims...)
    # out = zeros(T, 2, dims...)
    for idx in CartesianIndices(axes(out)[2:end])
        out[:, idx] .= rand(rng, sampler)
    end
    return out
end


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
- `kws...`: Keyword arguments `kws...` are passed to `QuadGK.quadgk` (for `model::ConstantMassFunction`) or `HCubature.hcubature` (for `model::RedshiftMassFunction`) to perform the numerical integration. These methods have default tolerances that are very tight `e.g., sqrt(eps)`, so loosening these tolerances (e.g., by setting relative tolerance `rtol=1e-4`)can significantly speed up the integration at the cost of some accuracy.

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
    @argcheck z1 < z2 "z1 must be less than z2"
    # integrate over z: N = ∫_{z1}^{z2} φcum(z) * dV/dz * dz
    integrand(x) = begin
        logMstar, z = x
        N = model(exp10(logMstar), z)
        N * ustrip(Mpc^3, comoving_volume_element(cosmo, z))
    end
    total = hcubature(integrand, (log10(mmin), z1), (log10(mmax), z2); kws...)[1]
    # total = Cubature.hcubature(integrand, (log10(mmin), z1), (log10(mmax), z2); kws...)[1] # Little slower than HCubature.hcubature
    # total = Cubature.pcubature(integrand, (log10(mmin), z1), (log10(mmax), z2); kws...)[1] # 20x slower than hcubature
    # Comoving volume element is per steradian, so multiply by 4π to get full sky
   return total * 4 * π
end


################################################
# Include specific mass function implementations
include("Schechter.jl")
export SchechterMassFunction, DoubleSchechterMassFunction
include("BinnedRedshiftMassFunction.jl")

end # module