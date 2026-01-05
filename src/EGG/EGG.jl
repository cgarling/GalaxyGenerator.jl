# EGG uses separate double Schechter mass functions for the star-forming galaxy (SFG) population and the quiescent galaxy (QG) population, so we separate implementation based on whether the *galaxy* as a whole is quiescent or star-forming. Within these classifications, galaxies can also vary in whether their bulges are SF or not

"""
Contains code to generate galaxy catalogs using methods similar to those used in the Empirical Galaxy Generator (Schreiber2017)
"""
module EGG

using ..GalaxyGenerator: interp_lin, interp_log, merge_add, find_bin, f_sky
using ..GalaxyGenerator.IGM: IGMAttenuation, transmission, tau, Inoue2014
using ..GalaxyGenerator.MassFunctions: RedshiftMassFunction, RedshiftMassFunctionSampler, MassFunctionSampler, BinnedRedshiftMassFunction, DoubleSchechterMassFunction, integrate

using ArgCheck: @argcheck, @check
using Cosmology: AbstractCosmology, angular_diameter_dist, distmod, Planck18
using Distributions: LogNormal, Normal, Uniform, Poisson
using DustExtinction: ExtinctionLaw, CCM89
using FITSIO: FITS
using IrrationalConstants: logten
import Logging
using PhotometricFilters: AbstractFilter, magnitude, MagnitudeSystem, mean_flux_density, zeropoint_mag, detector_type, wavelength, throughput
using Pkg.Artifacts: @artifact_str
using Random: Random, default_rng, AbstractRNG
using SpecialFunctions: erf
using StaticArrays: SVector
using Statistics: mean
import Unitful as u
import UnitfulAstro as ua

export egg, generate_galaxies

include("optlib.jl")
# Load default optical SED library
const optlib = OptLib()
const irlib = CS17_IRLib()

"""
A `BinnedRedshiftMassFunction` implementing the stellar mass functions used in EGG [Schreiber2017](@cite) for *star-forming* galaxies, see their Table A.1. Rather than using a piecewise constant mass function between redshift bins, we interpolate linearly in redshift between the two nearest mass functions.
"""
const EGGMassFunction_SF = BinnedRedshiftMassFunction(
    # [0.3, 0.7, 1.2, 1.8, 2.5, 3.5, 4.5],
    [(0.7 + 0.3)/2, (1.2 + 0.7)/2, (1.8 + 1.2)/2, (2.5 + 1.8)/2, (3.5 + 2.5)/2, (4.5 + 3.5)/2],
    [
        DoubleSchechterMassFunction(8.9e-4, -1.4, 1e11, 8.31e-5, 0.5, exp10(10.64)),
        DoubleSchechterMassFunction(7.18e-4, -1.4, 1e11, 4.04e-4, 0.5, exp10(10.73)),
        DoubleSchechterMassFunction(4.66e-4, -1.5, 1e11, 4.18e-4, 0.5, exp10(10.67)),
        DoubleSchechterMassFunction(2.14e-4, -1.57, 1e11, 4.06e-4, 0.5, exp10(10.84)),
        DoubleSchechterMassFunction(2.12e-4, -1.6, 1e11, 9.07e-5, 0.5, exp10(10.94)),
        DoubleSchechterMassFunction(4.45e-5, -1.7, 1e11, 8.6e-6, 0.5, exp10(11.69))
    ], true)

"""
A `BinnedRedshiftMassFunction` implementing the stellar mass functions used in EGG [Schreiber2017](@cite) for *quiescent* galaxies, see their Table A.1. Rather than using a piecewise constant mass function between redshift bins, we interpolate linearly in redshift between the two nearest mass functions.
"""
const EGGMassFunction_Q = BinnedRedshiftMassFunction(
    # [0.3, 0.7, 1.2, 1.8, 2.5, 3.5, 4.5],
    [(0.7 + 0.3)/2, (1.2 + 0.7)/2, (1.8 + 1.2)/2, (2.5 + 1.8)/2, (3.5 + 2.5)/2, (4.5 + 3.5)/2],
    [
        DoubleSchechterMassFunction(7.77e-5, -1.65, 1e11, 1.54e-3, -0.48, exp10(11.04)),
        DoubleSchechterMassFunction(3.54e-5, -1.6, 1e11, 1.04e-3, 0.06, exp10(10.86)),
        DoubleSchechterMassFunction(2.3e-5, -1.25, 1e11, 6.25e-4, 0.3, exp10(10.83)),
        DoubleSchechterMassFunction(1e-5, -1.0, 1e11, 1.73e-4, -0.17, exp10(11.05)),
        DoubleSchechterMassFunction(0.0, -1.0, 1e11, 1.22e-4, -0.26, exp10(10.94)),
        DoubleSchechterMassFunction(0.0, -1.0, 1e11, 3e-5, -0.3, 1e11)
    ], true)


"""
    get_mass_limit(z, SF::Bool, mag_lim, filt::AbstractFilter, mag_sys::MagnitudeSystem; 
                   Mstar=logrange(1e6, 1e13; length=100), cosmo::AbstractCosmology=Planck18, optlib::OptLib=optlib,
                   igm::IGMAttenuation=Inoue2014)

Given a redshift `z`, whether the galaxy is star-forming (`SF::Bool`), an apparent magnitude limit `mag_lim` in the filter `filt` with magnitude system `mag_sys`, returns the stellar mass limit `Mstar` (in M⊙) required to reach that magnitude limit.

# Arguments
- `z`: Redshift
- `SF::Bool`: Whether the galaxy is star-forming (`true`) or quiescent (`false`)
- `mag_lim`: Apparent magnitude limit in the specified filter and magnitude system
- `filt::AbstractFilter`: Photometric filter
- `mag_sys::MagnitudeSystem`: Magnitude system for the filter

# Keyword Arguments
- `Mstar`: Range of stellar masses (in M⊙) to sample when calculating the mass limit. The returned mass limit will be constrained to be within this range.
*Must be sorted ascending.*
- `cosmo::AbstractCosmology`: Cosmology to use for distance modulus calculations
- `optlib::OptLib`: Optical SED library to use for generating galaxy SEDs
- `igm::IGMAttenuation`: IGM attenuation model to use when generating galaxy SEDs

# Returns
- `mass_lim`: Stellar mass limit (in M⊙) required to reach the specified magnitude limit at redshift `z`

Below we calculate the stellar mass limit at redshift 2 for star-forming galaxies with an apparent magnitude limit of 25 in the HST/ACS F606W filter in the Vega magnitude system.

```jldoctest
julia> using GalaxyGenerator.EGG: get_mass_limit

julia> using PhotometricFilters: HST_ACS_WFC_F606W, Vega

julia> result = get_mass_limit(2.0, true, 26.0, HST_ACS_WFC_F606W(), Vega());

julia> isapprox(result, 1.077217e8; rtol=1e-2)
true
```
"""
function get_mass_limit(z, SF::Bool, mag_lim, filt::AbstractFilter, mag_sys::MagnitudeSystem; kws...)
    return get_mass_limit(z, SF, mag_lim, filt, zeropoint_mag(filt, mag_sys); kws...) # Calculate zeropoint_mag and call below function
end
function get_mass_limit(z, SF::Bool, mag_lim, filt::AbstractFilter, zpt; 
    Mstar=logrange(1e6, 1e13; length=100), cosmo::AbstractCosmology=Planck18, optlib::OptLib=optlib,
    igm::IGMAttenuation=Inoue2014)
    # Generate Mstar - flux relation
    m2l_cor = get_m2l_cor(z) # M/L correction in dex
    dmod = distmod(cosmo, z)
    # zpt = zeropoint_mag(filt, mag_sys)

    mags = Vector{Float64}(undef, length(Mstar))
    for i in eachindex(Mstar)
        # Get UV, VJ colors, optical SED
        uv, vj = uv_vj(log10(Mstar[i]), z, SF; rng=nothing)
        λ, sed = get_opt_sed(uv, vj, optlib)
        # Convert SED to units of erg Å^-1 cm^-2 s^-1
        sed .*= exp10(log10(Mstar[i]) - m2l_cor)
        # Add IGM attenuation; this takes most of the runtime
        @. sed *= transmission(igm, z, λ)
        λ .*= 1 + z # Redshift wavelengths
        # mags[i] = magnitude(filt, mag_sys, λ * u.μm, sed * u.erg / u.s / u.cm^2 / u.angstrom) + dmod
        fbar = mean_flux_density(λ, sed, filt.(λ), detector_type(filt))
        mags[i] = magnitude(fbar, zpt) + dmod
    end
    
    min_mag, max_mag = extrema(mags)
    if mag_lim > max_mag
        @warn "The minimum stellar mass $(first(Mstar)) provided to `get_mass_limit` for redshift $z is too high to sample galaxies to the requested magnitude limit $mag_lim. Returning $(first(Mstar))."
        return Mstar[1]
    elseif mag_lim < min_mag
        @warn "The maximum stellar mass $(last(Mstar)) provided to `get_mass_limit` for redshift $z is too low to sample galaxies brighter than the requested magnitude limit $mag_lim. Returning $(last(Mstar))."
        return Mstar[end]
    else
        # Assumes monotonically decreasing mags with Mstar, this is true for low mass galaxies
        # but breaks down for higher masses (>1e10)
        # mass_lim = Mstar[searchsortedfirst(mags, mag_lim; rev=true)]
        # mass_lim = interp_log(reverse(mags), reverse(Mstar), mag_lim)
        # Mstar sorted from low mass to high mass -> just find first entry where mags <= mag_lim
        mass_lim = Mstar[findfirst(<=(mag_lim), mags)]
        mass_lim /= 2 # Add safety factor of 2
        return max(first(Mstar), mass_lim) # Don't go below minimum Mstar provided
    end
end

"""
    compute_magnitude(filt, zpt, λ, sed; dmod=0)
More efficient version of `PhotometricFilters.magnitude` that uses a precomputed zeropoint `zpt`. `dmod` is distance modulus added to the magnitude."""
function compute_magnitude(filt, zpt, λ, sed; dmod=0)
    @argcheck length(λ) == length(sed) "λ and sed must have the same length"
    T = promote_type(eltype(λ), eltype(sed)) # Only promote up to λ, sed precision
    good = (u.ustrip(u.angstrom, first(wavelength(filt))) .<= λ) .& (λ .<= u.ustrip(u.angstrom, last(wavelength(filt))))
    λ_good = view(λ, good)
    fbar = mean_flux_density(λ_good, view(sed, good), filt.(λ_good), detector_type(filt))
    return T(magnitude(fbar, zpt) + dmod)
end
# This is inefficient as `magnitude` resamples the filter to the λ every time it's called; don't care for now
# mag_abs = [magnitude(filters[i], mag_sys[i], λ * u.angstrom, sed * u.erg / u.s / u.cm^2 / u.angstrom) for i in eachindex(filters)]
# mag_abs = [magnitude(mean_flux_density(filters[i], λ, sed), zpts[i]) for i in eachindex(filters)]

"""
    uv_vj(logMstar, z, SF::Bool)
Takes `log10(Mstar [M⊙])`, redshift, and `SF::Bool` determining whether the stellar population is star-forming (`true`) or quiescent (`false`).

Returns U-V and V-J colors `(uv, vj)`.
"""
# function uv_vj1(logMstar, z, SF::Bool; rng::AbstractRNG=default_rng())
#     # This implementation is based on description in paper
#     return if SF
#         a0 = 0.48 * erf(logMstar - 10) + 1.15
#         a1 = -0.28 + 0.25 * max(0, logMstar - 10.35)
#         vj = a0 + a1 * min(z, 3.3) # V-J color for star-forming galaxies
#         vj = min(vj, 1.7) # limit to <1.7
#         vj = rand(rng, Normal(vj, 0.1)) # Add first error
#         uv = 0.65 * vj + 0.45
#         vj = rand(rng, Normal(vj, 0.12)) # Add extra error
#         uv = rand(rng, Normal(uv, 0.12)) # Add extra error
#         (uv, vj)
#     else
#         vj = 0.1 * (logMstar - 11) + 1.25
#         vj = rand(rng, Normal(vj, 0.1)) # Add first error
#         vj = max(min(vj, 1.45), 1.15) # Restrict 1.15 <= V-J <= 1.45
#         uv = 0.88 * vj + 0.75
#         vj = rand(rng, Normal(vj, 0.1)) # Add extra error
#         uv = rand(rng, Normal(uv, 0.1)) # Add extra error
#         (uv, vj)
#     end
# end
function uv_vj(logMstar, z, SF::Bool; rng::Union{AbstractRNG,Nothing}=default_rng())
    # This implementation is based on the code in the egg repository
    uv, vj = if SF
        a0 = 0.58 * erf(logMstar - 10) + 1.39
        a1 = -0.34 + 0.3 * max(0, logMstar - 10.35)
        a = a0 + a1 * min(z, 3.3)
        a = min(a, 2.0)
        rnd_amp = 0.2 + (0.25 + 0.12 * clamp((z - 0.5) / 2.0, 0.0, 1.0)) *
                max(1.0 - 2.0*abs(logMstar - (10.3 + 0.4 * erf(z - 1.5))), 0.0)
        a = isnothing(rng) ? a : rand(rng, Normal(a, rnd_amp))

        # Move in UVJ diagram according to UVJ vector
        slope = 0.65
        theta = atan(slope) # This can be simplified but don't care right now
        vj = 0.0 + a * cos(theta)
        uv = 0.45 + a * sin(theta)
        vj = isnothing(rng) ? vj : rand(rng, Normal(vj, 0.15))
        uv = isnothing(rng) ? uv : rand(rng, Normal(uv, 0.15))
        (uv, vj)
    else
        vj = 0.1 * (logMstar - 11) + 1.25
        vj = isnothing(rng) ? vj : rand(rng, Normal(vj, 0.1))
        vj = max(min(vj, 1.45), 1.15) # Restrict 1.15 <= V-J <= 1.45
        uv = 0.88 * vj + 0.6
        vj = isnothing(rng) ? vj : rand(rng, Normal(vj, 0.1)) # Add extra error
        uv = isnothing(rng) ? uv : rand(rng, Normal(uv, 0.1)) # Add extra error
        (uv, vj)
    end
    # Add additional color offset depending on redshift
    uv += 0.4 * max((0.5 - z) / 0.5, 0.0)
    vj += 0.2 * max((0.5 - z) / 0.5, 0.0)
    return uv, vj
end

# SF is whether galaxy is star-forming or not
# This implementation just generates simple galaxy properties
# This method is called by the method below that also takes
# PhotometricFilters and computes magnitudes 
function egg(Mstar, z, SF::Bool; 
    rng::Union{Nothing,AbstractRNG}=default_rng())

    # Mstar, z = Float32(Mstar), Float32(z)
    logMstar = log10(Mstar)
    log1pz = log1p(z) / logten # same as log10(z + 1)
    # Distributions for the bulge-to-total mass ratio
    BT = if SF
        exp10(-0.7 + 0.27 * (logMstar - 10))
    else
        exp10(-0.3 + 0.1 * (logMstar - 10))
    end
    BT = isnothing(rng) ? BT : rand(rng, LogNormal(log(BT), 0.2 * logten))
    BT = clamp(BT, 0.0, 1.0)
    Mbulge = Mstar * BT
    Mdisk = Mstar - Mbulge
    if isnan(Mbulge)
        println(Mstar, " ", BT)
    end
    @check Mbulge >= 0
    @check Mdisk >= 0

    # Distributions for bulge, disk sizes
    Fz_disk = if z <= 1.7
        0.41 - 0.22 * log1pz
    else
        0.62 - 0.7 * log1pz
    end
    Fz_bulge = if z <= 0.5
        0.78 - 0.6 * log1pz
    else
        0.9 - 1.3 * log1pz
    end
    R50_disk = exp10(0.2 * (min(10.6, logMstar) - 9.35) + Fz_disk) # kpc
    R50_disk = isnothing(rng) ? R50_disk : rand(rng, LogNormal(log(R50_disk), 0.17 * logten))
    R50_bulge = exp10(0.56 * (logMstar - 11.25) + Fz_bulge) # kpc
    R50_bulge = isnothing(rng) ? R50_bulge : rand(rng, LogNormal(log(R50_bulge), 0.2 * logten))
    α_R = max(0.0, 1 - 0.8 * log10(R50_disk / R50_bulge))
    BT_α = BT^α_R
    R50_tot = R50_disk * (1 - BT_α) + R50_bulge * BT_α
    # Uniform position angle, shared between bulge and disk
    PA = isnothing(rng) ? 0.0 : rand(rng, Uniform(-180, 180))
    # Bulge and disk axis ratios, b/a
    # Bulges: calibration from n>2.5 galaxies and M* > 10.5
    bulge_ratio = if isnothing(rng)
        0.615
    else
        cdf = SVector(0.0, 0.0, 0.005641797168843602, 0.046194351364289135, 0.1283252410586063, 0.24369144498392944, 0.3792997332968612, 0.5338507830130617, 0.6942145934486769, 0.8460986117759693, 0.9627641386856322, 1.0)
        x = SVector(0.0, 0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0)
        interp_lin(cdf, x, rand(rng))
    end
    # Disks: calibration from n<1.5 galaxies and M* > 9.0
    disk_ratio = if isnothing(rng)
        0.523
    else
        cdf = SVector(0.0, 0.0, 0.011221453411250134, 0.09819668017065215, 0.2446850464274191, 0.4074498978238269, 0.5641917326927904, 0.7045136772666979, 0.8244720897716273, 0.9193346000788728, 0.9814290323737138, 1.0)
        x = SVector(0.0, 0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0)
        interp_lin(cdf, x, rand(rng))
    end

    disk_SF = SF # Disks are SF if galaxy is SF
    bulge_SF = if ~SF # Bulge is quiescent if galaxy is quiescent
        false
    else
        if BT >= 0.6 # Bulge is quiescient if galaxy is bulge-dominated
            false
        else
            isnothing(rng) ? false : rand(rng, (true, false)) # Otherwise, 50% chance of quiescent/SF
        end
    end

    # SFRs, initially in log10(sfr) unit
    ms_disp = 0.3 # Dispersion of SFMS in dex; EGG notes this can change quality of simulation
    sfrms = logMstar - 9.67 + 1.82 * log1pz - 0.38 * max(0.0, logMstar - 9.59 - 2.22 * log1pz)^2
    sfr = if SF 
        isnothing(rng) ? sfrms : rand(rng, Normal(sfrms, ms_disp))
    else # If passive,
        sfr_q = min(sfrms, 0.5 * (logMstar - 11) + log1pz - 0.6)
        isnothing(rng) ? sfr_q : rand(rng, Normal(sfr_q, 0.4))
        # rand(rng, Normal(min(sfrms, 0.5 * (logMstar - 11) + log1pz - 0.6), 0.4))
    end
    # Add starburst, 3.3% of star-forming galaxies
    if SF && !isnothing(rng) && rand(rng) < 0.033
        sfr += 0.72 # Enhance SFR by a factor of 5.24
    end
    rsb = sfr - sfrms # Ratio of SFR / SFR_MS
    sfr = exp10(sfr) # Convert to linear units

    # IRX = log10(L_IR / L_UV)
    irx = (0.45 * min(z, 3.0) + 0.35) * (logMstar - 10.5) + 1.2
    irx = exp10(isnothing(rng) ? irx : rand(rng, Normal(irx, 0.4)))
    sfr_ir = sfr / (1 + 1 / irx)
    sfr_uv = sfr / (1 + irx)

    # Metallicity
    # oh = 9.07
    # mu32 = (logMstar - 0.2) - 0.32 * (log10(sfr) - 0.2)
    # oh = mu32 < 10.36 ? 8.9 + 0.47 * (mu32 - 10) : oh
    # oh -= 8.69 # In solar metallicity
    # Broken FMR (Bethermin+16), tweaked to reproduce [OIII]/Hbeta @ z=2 (Dickey+16)
    # oh = z > 1 ? oh - 0.2 * clamp(z - 1, 0.0, 1.0) : oh
    # oh = clamp(oh, 8.69 - 0.6, 8.69 + 1.0)

    # Use Jain2025 redshift-dependent model
    # _M0 = 9.72 + 2.63 * (z + 1) # Redshift-dependent turnover mass
    # oh = 8.78 - 0.25 / 1.2 * log1p((Mstar / _M0)^-1.2) / logten
    
    # Use Sanders2021 redshift-dependent FMR model
    μα = logMstar - 0.6 * log10(sfr) # Equation 9
    _y = μα - 10
    _y2 = _y^2
    oh = 8.8 + 0.188 * _y - 0.22 * _y2 - 0.0531 * _y2 * _y

    # Optical colors
    uv_disk, vj_disk = uv_vj(logMstar, z, disk_SF; rng=rng)
    uv_bulge, vj_bulge = uv_vj(logMstar, z, bulge_SF; rng=rng)

    # IR properties
    tdust_ms = 32.13 + 4.6 * (z - 2) # T_dust main sequence value at z
    tdust = tdust_ms + 10.1 * rsb # Starbursts are warmer
    # Massive galaxies are colder (=downfall of SFE)
    tdust -= 1.5 * max(0.0, 2 - z) * clamp(logMstar - 10.7, 0.0, 1.0)
    tdust = isnothing(rng) ? tdust : rand(rng, Normal(tdust, 0.12 * tdust_ms))
    # Ratio of IR to 8 μm luminosity (Elbaz 2011)
    ir8 = log10((4.08 + 3.29 * clamp(z - 1, 0.0, 1.0)) * 0.81)
    ir8 += 0.66 * max(0.0, rsb) # Starbursts have higher ir8
    ir8 += -clamp(logMstar - 10, -1.0, 0.0) # low-mass galaxies have larger ir8
    ir8 = exp10(isnothing(rng) ? ir8 : rand(rng, Normal(ir8, 0.18)))
    ir8 = clamp(ir8, 0.48, 27.5) # range allowed in IR library
    lir = sfr_ir / 1.72e-10 # Infrared luminosity from 8 to 1000 μm in L⊙
    fpah = clamp(1 / (1 - (331 - 691 * ir8) / (193 - 6.98 * ir8)), 0.0, 1.0)
    lir = isfinite(lir) ? lir : 0.0

    return (Mstar = Mstar, z = z, sfr = sfr, sfr_ir = sfr_ir, sfr_uv = sfr_uv, uv_disk = uv_disk, vj_disk = vj_disk, uv_bulge = uv_bulge, vj_bulge = vj_bulge, R50_disk = R50_disk, R50_bulge = R50_bulge, R50 = R50_tot, disk_ratio = disk_ratio, bulge_ratio = bulge_ratio, PA = PA, BT = BT, Mdisk = Mdisk, Mbulge = Mbulge, Tdust = tdust, fpah = fpah, lir = lir, ir8 = ir8, IRX = irx, OH = oh)
end

# This function is basically and intermediary between above function and below function.
# It computes basic properties using the function above, which returns a NamedTuple,
# then it feeds that NamedTuple into the function below that computes magnitudes.
function egg(Mstar, z, SF::Bool, 
    @nospecialize(filters::AbstractVector{<:AbstractFilter}),
    @nospecialize(mag_sys::AbstractVector{<:MagnitudeSystem});
    rng::Union{Nothing,AbstractRNG}=default_rng(),
    kws...)

    zpts = zeropoint_mag.(filters, mag_sys)
    return egg(Mstar, z, SF, filters, zpts; rng, kws...) # Call below ...
end
function egg(Mstar, z, SF::Bool, 
    @nospecialize(filters::AbstractVector{<:AbstractFilter}),
    zpts::AbstractVector;
    rng::Union{Nothing,AbstractRNG}=default_rng(),
    kws...)

    # return egg(egg(Mstar, z, SF; rng=rng), filters, zpts; rng=rng, kws...)
    r = egg(Mstar, z, SF; rng=rng)
    return egg(r, filters, zpts; rng=rng, kws...)
end

function egg(
    r::NamedTuple, 
    @nospecialize(filters::AbstractVector{<:AbstractFilter}),
    # @nospecialize(mag_sys::AbstractVector{<:MagnitudeSystem});
    zpts::AbstractVector;
    cosmo::AbstractCosmology=Planck18,
    rng::Union{Nothing,AbstractRNG}=default_rng(), 
    optlib::OptLib=optlib,
    irlib::IRLib=irlib,
    igm::IGMAttenuation=Inoue2014,
    extinction_law::ExtinctionLaw=CCM89(Rv=3.1),
    Av::Number=0.0 # Foreground MW extinction in V-band magnitude
    )

    @argcheck length(filters) == length(zpts) # length(mag_sys)
    @argcheck Av >= 0 "V-band extinction Av must be non-negative."
    Mstar, z = r.Mstar, r.z
    logMstar = log10(Mstar)
    dmod = distmod(cosmo, z) # Cosmological distance modulus at redshift z

    # Compute sizes in arcsec from kpc in original result
    propsize = u.ustrip(ua.kpc, angular_diameter_dist(cosmo, z)) # in kpc / radians
    propsize = propsize * π / (180 * 3600) # in kpc / arcsec
    R50_disk_arcsec = r.R50_disk / propsize
    R50_bulge_arcsec = r.R50_bulge / propsize
    R50_arcsec = r.R50 / propsize

    # Get optical SEDs; SEDs returned in units of erg/s/cm^2/Å at 10 pc per unit stellar mass
    m2l_cor = get_m2l_cor(z) # M/L correction in dex
    λ_disk, sed_disk, Av_disk = get_opt_sed(r.uv_disk, r.vj_disk, optlib)
    λ_bulge, sed_bulge, Av_bulge = get_opt_sed(r.uv_bulge, r.vj_bulge, optlib)
    sed_disk .*= exp10(log10(r.Mdisk) - m2l_cor)
    sed_bulge .*= exp10(log10(r.Mbulge) - m2l_cor)

    # Get IR SED
    ir_result = get_ir_sed(r.Tdust, irlib)
    ir_λ = ir_result.lam # angstroms

    # Correct fpah, if necessary
    fpah = if typeof(irlib) == CS17_IRLib
        # r.lir / (ir_result.lir_dust * (1 - r.fpah) + ir_result.lir_pah * r.fpah)
        clamp(1.0 / (1.0 - (ir_result.lir_pah - ir_result.l8_pah * r.ir8) /
            (ir_result.lir_dust - ir_result.l8_dust * r.ir8)), 0.0, 1.0)
    else
        r.fpah # For some reason, EGG sets this to 0.04 despite having calculated it above
    end

    # Calculate dust mass
    Mdust = if typeof(irlib) == CS17_IRLib
        r.lir / (ir_result.lir_dust * (1 - fpah) + ir_result.lir_pah * fpah)
    else
        r.lir / 1e3
    end

    ir_sed = if typeof(irlib) == CS17_IRLib
        # ir_sed is returned from get_ir_sed in erg/s/cm^2/Å at 10 pc per unit dust mass
        dust, pah = ir_result.dust, ir_result.pah
        @. Float32((dust * (1 - fpah) + pah * fpah) * Mdust)
    else
        ir_result.sed # Assumes SED returned from get_ir_sed is in erg/s/cm^2/Å at 10 pc already
    end

    ################
    # Gas properties
    # oh_solar = r.OH - 8.69 # in units of solar metallicity
    # gdr = 2.23 - oh_solar  # gas-to-dust ratio too low here
    # log10(gas-to-dust ratio) from Rémy-Ruyer+14
    gdr = r.OH > 7.96 ? 2.21 + 1.0 * (8.69 - r.OH) : 0.96 + 3.08 * (8.69 - r.OH)
    gdr = isnothing(rng) ? gdr : rand(rng, Normal(gdr, 0.04))
    Mgas = Mdust * exp10(gdr) # gas mass in M⊙
    MH2 = Mgas * 0.3 # assume 30% of gas is molecular
    MH2 = isnothing(rng) ? MH2 : rand(rng, LogNormal(log(MH2), 0.2 * logten))

    #########################
    # Generate emission lines
    # Attenuation of lines (Pannella+15 for redshift dependence)
    Avlines_disk = (log10(r.sfr / r.sfr_uv) * 0.95 + Av_disk) / 2
    Avlines_disk *= interp_lin(SVector(0.0, 1.0, 2.0, 100.0), SVector(1.7, 1.3, 1.0, 1.0), z)
    Avlines_disk = isnothing(rng) ? Avlines_disk : rand(rng, Normal(Avlines_disk, 0.1))
    Avlines_bulge = isnothing(rng) ? Av_bulge : rand(rng, Normal(Av_bulge, 0.1))
    Avlines_disk = clamp(Avlines_disk, 0.0, 6.0)
    Avlines_bulge = clamp(Avlines_bulge, 0.0, 6.0)
    # Ly-α escape fraction, Hayes+10
    fescape_disk = log10(0.445) - 0.4 * Av_disk / 4.05 * 17.8
    # Correction to avoid over-producing Ly-α at z~1-2
    fescape_disk += interp_lin(SVector(0.5, 1.2, 2.2, 3.0, 4.0), SVector(-1.2, -1.2, -0.8, 0.0, 0.0), z; extrapolate=true)
    fescape_disk = isnothing(rng) ? exp10(fescape_disk) : rand(rng, LogNormal(fescape_disk * logten, 0.4 * logten))
    fescape_disk = clamp(fescape_disk, 0.0, 1.0)

    # Velocity dispersion, Stott+16
    vdisp = exp10(0.12 * (logMstar - 10) + 1.78)


    #####################
    # Add emission lines
    # Merge optical and IR SEDs
    opt_λ, opt_sed = merge_add(λ_bulge, λ_disk, sed_bulge, sed_disk)
    λ, sed = merge_add(opt_λ, ir_λ, opt_sed, ir_sed)

    # Obtain rest-frame absolute magnitudes
    mag_abs = compute_magnitude.(filters, zpts, Ref(λ), Ref(sed))

    # Apply IGM absorption, uses rest-frame wavelengths
    τ = tau.(igm, z, λ) # This is working, but expensive ~ 1 ms for full SED
    # Add MW extinction, requires observer-frame wavelengths
    λ .*= 1 + z # Redshift wavelengths to observer frame
    # extinction_law(λ) returns A(λ) / Av, convert to τ
    if Av != 0
        @. τ += extinction_law(λ) * Av / (2.5 * log10(ℯ)) # The constant is ~1.086
    end
    @. sed *= exp(-τ)

    # Measure observed magnitudes
    mag_obs = compute_magnitude.(filters, zpts, Ref(λ), Ref(sed); dmod=dmod)

    return merge(r, (R50_arcsec = R50_arcsec, R50_bulge_arcsec = R50_bulge_arcsec, R50_disk_arcsec = R50_disk_arcsec, vdisp = vdisp, Mgas = Mgas, MH2 = MH2, fpah = fpah, Mdust = Mdust, mag_abs = mag_abs, mag_obs = mag_obs, λ = λ, sed = sed))
    # mag_obs = [-25//10 * log10(mean_flux_density(filters[i], λ, sed)) - zeropoint_mag(filters[i], mag_sys[i]) for i in eachindex(filters)]
    # function _itp(f::AbstractFilter, λ)
    #     wave = wavelength(f)
    #     λv = @view λ[searchsortedfirst(λ, first(wave)):searchsortedlast(λ, last(wave))]

    # mag_obs = [-25//10 * log10(mean_flux_density(λ, sed, [interp_lin(wavelength(filters[i]), throughput(filters[i]), j * u.angstrom) for j in λ], detector_type(filters[i]))) - zeropoint_mag(filters[i], mag_sys[i]) for i in eachindex(filters)]

end

#################################################################################
# `generate_galaxies` here will return bulk galaxy properties but *no* photometry

# This will generate `RedshiftMassFunctionSampler`s for SF and Q galaxies from provided `sf_massfunc` and `q_massfunc` keyword arguments and call below function.
# For small number of galaxies, runtime is dominated by constructing the samplers, not running egg
function generate_galaxies(mmin, mmax, zmin, zmax, area_deg2, args...;
    cosmo::AbstractCosmology=Planck18,
    q_massfunc::RedshiftMassFunction=EGGMassFunction_Q,
    sf_massfunc::RedshiftMassFunction=EGGMassFunction_SF,
    npoints_mass::Int=100, npoints_redshift::Int=100,
    kws...)

    @argcheck mmin < mmax "Minimum stellar mass `mmin` must be less than maximum stellar mass `mmax`."
    @argcheck zmin < zmax "Minimum redshift `zmin` must be less than maximum redshift `zmax`."
    sf_sampler = MassFunctionSampler(sf_massfunc, cosmo, mmin, mmax, zmin, zmax; npoints_mass, npoints_redshift)
    q_sampler = MassFunctionSampler(q_massfunc, cosmo, mmin, mmax, zmin, zmax; npoints_mass, npoints_redshift)

    return generate_galaxies(sf_sampler, q_sampler, area_deg2, args...; cosmo, kws...)
end

function generate_galaxies(
    sf_sampler::RedshiftMassFunctionSampler,
    q_sampler::RedshiftMassFunctionSampler, 
    # mmin, mmax, zmin, zmax, area_deg2;
    area_deg2::Number;
    cosmo::AbstractCosmology=Planck18,
    rng::AbstractRNG=default_rng(),
    use_rng::Bool=true,  # We *have* to use random sampling for redshifts, stellar masses, but RNG for galaxy properties is optional
    poisson::Bool=false) # Whether to add Poisson scatter to number of galaxies (true) or just return expected number (false)

    @argcheck 0 < area_deg2 < 4π * (180/π)^2 "Area in deg² must be between 0 and the full sky (~41253 deg²)."
    # Retrieve mass and redshift limits from samplers
    mmin_sf, mmax_sf = sf_sampler.mass_grid[1], sf_sampler.mass_grid[end]
    zmin_sf, zmax_sf = sf_sampler.redshift_grid[1], sf_sampler.redshift_grid[end]
    mmin_q, mmax_q = q_sampler.mass_grid[1], q_sampler.mass_grid[end]
    zmin_q, zmax_q = q_sampler.redshift_grid[1], q_sampler.redshift_grid[end]

    # Calculate expected number of galaxies with masses between mmin and mmax and redshifts between zmin and zmax
    N_sf = integrate(sf_sampler, cosmo, mmin_sf, mmax_sf, zmin_sf, zmax_sf) * f_sky(area_deg2)
    N_q = integrate(q_sampler, cosmo, mmin_q, mmax_q, zmin_q, zmax_q) * f_sky(area_deg2)

    # Poisson sample, if requested
    N_sf = poisson ? rand(rng, Poisson(N_sf)) : round(Int, N_sf)
    N_q = poisson ? rand(rng, Poisson(N_q)) : round(Int, N_q)

    @info "Number of star-forming galaxies: $N_sf"
    @info "Number of quiescent galaxies: $N_q"

    # TODO: Even if N_sf and N_q are very large, break them up into chunks of some smaller number of galaxies,
    # sample the stellar masses and redshifts, and then cull the lists using `get_mass_limit` to avoid memory issues.
    N_sf > 1e7 && @warn "The expected number of star-forming galaxies ($N_sf) is very large; this may take a long time and use a lot of memory or crash outright."
    N_q > 1e7 && @warn "The expected number of quiescent galaxies ($N_q) is very large; this may take a long time and use a lot of memory or crash outright."

    # Sample redshifts and stellar masses
    sf = rand(rng, sf_sampler, N_sf) # 2 x N_sf array of (Mstar, z)
    q = rand(rng, q_sampler, N_q)    # 2 x N_q array of (Mstar, z)

    rng = use_rng ? rng : nothing
    # Get first result so we know the output return type
    r1 = egg(sf[1, 1], sf[2, 1], true; rng=rng)
    results = Vector{typeof(r1)}(undef, N_sf + N_q)
    results[1] = r1
    Threads.@threads for i in 1:N_sf
        results[i] = egg(sf[1, i], sf[2, i], true; rng=rng)
    end
    Threads.@threads for i in 1:N_q
        results[N_sf + i] = egg(q[1, i], q[2, i], false; rng=rng)
    end

    return results
end

#################################################################################
# `generate_galaxies` here will return bulk galaxy properties *and* photometry, requiring input filters and mag_sys
# Computes magnitude zeropoint and calls below function
# function generate_galaxies(mmin, mmax, zmin, zmax, area_deg2, mag_lim, mag_lim_idx::Int, filters::AbstractVector{<:AbstractFilter}, mag_sys::AbstractVector{<:MagnitudeSystem}; kws...)
#     zpt = zeropoint_mag(filters[mag_lim_idx], mag_sys[magl_lim_idx])
#     return generate_galaxies(mmin, mmax, zmin, zmax, area_deg2, mag_lim, filters[mag_lim_idx], zpt; kws...)
# end
# function generate_galaxies(mmin, mmax, zmin, zmax, area_deg2, mag_lim, filt::AbstractFilter, mag_sys::MagnitudeSystem; kws...)
#     zpt = zeropoint_mag(filt, mag_sys)
#     return generate_galaxies(mmin, mmax, zmin, zmax, area_deg2, mag_lim, filt, zpt; kws...)
# end
function generate_galaxies(mmin, mmax, zmin, zmax, area_deg2, mag_lim, mag_lim_idx::Int, filters::AbstractVector{<:AbstractFilter}, zpts::AbstractVector{<:MagnitudeSystem}; kws...)
    zpts = zeropoint_mag.(filters, zpts)
    return generate_galaxies(mmin, mmax, zmin, zmax, area_deg2, mag_lim, mag_lim_idx, filters, zpts; kws...)
end
# Here we construct the MassFunctionSamplers using the provided mag_lim, filter, and zeropoint
# This is actually kind of slow (~3s for npoints_mass=100, npoints_redshift=100) because of the calls to get_mass_limit,
# ends up computing 10,000 mock SEDs to get mass limits which is not always a good use of time if only a small number of galaxies are being generated
# function generate_galaxies(mmin, mmax, zmin, zmax, area_deg2, mag_lim, filt::AbstractFilter, zpt;
function generate_galaxies(mmin, mmax, zmin, zmax, area_deg2, mag_lim, mag_lim_idx::Int, filters::AbstractVector{<:AbstractFilter}, zpts::AbstractVector;
    cosmo::AbstractCosmology=Planck18,
    optlib::OptLib=optlib,
    igm::IGMAttenuation=Inoue2014,
    q_massfunc::RedshiftMassFunction=EGGMassFunction_Q,
    sf_massfunc::RedshiftMassFunction=EGGMassFunction_SF,
    npoints_mass::Int=100, npoints_redshift::Int=100,
    kws...)

    # Select out filter and zeropoint for computing the stellar mass limits
    filt = filters[mag_lim_idx]
    zpt = zpts[mag_lim_idx]

    @argcheck mmin < mmax "Minimum stellar mass `mmin` must be less than maximum stellar mass `mmax`."
    @argcheck zmin < zmax "Minimum redshift `zmin` must be less than maximum redshift `zmax`."

    redshift_grid = range(zmin, zmax, length=npoints_redshift)
    mstar_grid = logrange(mmin, mmax, length=npoints_mass)

    # Temporarily suppress warnings from get_mass_limit
    mmin_sf, mmin_q = Logging.with_logger(Logging.SimpleLogger(stderr, Logging.Error)) do
        mmin_sf = [get_mass_limit(z, true, mag_lim, filt, zpt; Mstar=mstar_grid, cosmo, optlib, igm) for z in redshift_grid]
        mmin_q = [get_mass_limit(z, false, mag_lim, filt, zpt; Mstar=mstar_grid, cosmo, optlib, igm) for z in redshift_grid]
        (mmin_sf, mmin_q)
    end

    sf_sampler = MassFunctionSampler(sf_massfunc, cosmo, mmin_sf, fill(mmax, npoints_mass), redshift_grid; npoints_mass)
    q_sampler = MassFunctionSampler(q_massfunc, cosmo, mmin_q, fill(mmax, npoints_mass), redshift_grid; npoints_mass)
    return generate_galaxies(sf_sampler, q_sampler, area_deg2, filters, zpts; cosmo, optlib, igm, kws...)
end
# If `mag_lim` isn't provided, we just construct the samplers and call below function
# Computes magnitude zeropoint and calls below function
function generate_galaxies(mmin, mmax, zmin, zmax, area_deg2, filters::AbstractVector{<:AbstractFilter}, 
                           mag_sys::AbstractVector{<:MagnitudeSystem}; kws...)
    zpts = zeropoint_mag.(filters, mag_sys)
    return generate_galaxies(mmin, mmax, zmin, zmax, area_deg2, filters, zpts; kws...)
end
function generate_galaxies(mmin, mmax, zmin, zmax, area_deg2, filters::AbstractVector{<:AbstractFilter}, zpts::AbstractVector;
    cosmo::AbstractCosmology=Planck18,
    q_massfunc::RedshiftMassFunction=EGGMassFunction_Q,
    sf_massfunc::RedshiftMassFunction=EGGMassFunction_SF,
    npoints_mass::Int=100, npoints_redshift::Int=100,
    kws...)

    sf_sampler = MassFunctionSampler(sf_massfunc, cosmo, mmin, mmax, zmin, zmax; npoints_mass, npoints_redshift)
    q_sampler = MassFunctionSampler(q_massfunc, cosmo, mmin, mmax, zmin, zmax; npoints_mass, npoints_redshift)

    # return sf_sampler, q_sampler
    # return which(generate_galaxies, typeof.((sf_sampler, q_sampler, area_deg2, filters, zpts)))
    return generate_galaxies(sf_sampler, q_sampler, area_deg2, filters, zpts; cosmo, kws...)
end
# Calculate zeropoint mags and call below function
function generate_galaxies(
    sf_sampler::RedshiftMassFunctionSampler, 
    q_sampler::RedshiftMassFunctionSampler, 
    area_deg2::Number, 
    @nospecialize(filters::AbstractVector{<:AbstractFilter}), 
    @nospecialize(mag_sys::AbstractVector{<:MagnitudeSystem}); 
    kws...)

    return generate_galaxies(sf_sampler, q_sampler, area_deg2, filters, zeropoint_mag.(filters, mag_sys); kws...)
end


function generate_galaxies(
    sf_sampler::RedshiftMassFunctionSampler,
    q_sampler::RedshiftMassFunctionSampler, 
    area_deg2::Number, 
    @nospecialize(filters::AbstractVector{<:AbstractFilter}),
    zpts::AbstractVector;
    cosmo::AbstractCosmology=Planck18,
    rng::AbstractRNG=default_rng(),
    use_rng::Bool=true,  # We *have* to use random sampling for redshifts, stellar masses, but RNG for galaxy properties is optional
    poisson::Bool=false, # Whether to add Poisson scatter to number of galaxies (true) or just return expected number (false)
    kws...)

    @argcheck 0 < area_deg2 < 4π * (180/π)^2 "Area in deg² must be between 0 and the full sky (~41253 deg²)."
    # Retrieve mass and redshift limits from samplers
    mmin_sf, mmax_sf = sf_sampler.mass_grid[1], sf_sampler.mass_grid[end]
    zmin_sf, zmax_sf = sf_sampler.redshift_grid[1], sf_sampler.redshift_grid[end]
    mmin_q, mmax_q = q_sampler.mass_grid[1], q_sampler.mass_grid[end]
    zmin_q, zmax_q = q_sampler.redshift_grid[1], q_sampler.redshift_grid[end]

    # Calculate expected number of galaxies with masses between mmin and mmax and redshifts between zmin and zmax
    # This will integrate using the analytic mass functions, not the samplers which may have different stellar mass limits
    # due to a pre-imposed magnitude limit (see mag_lim above). So the right way to do this is to integrate over the 
    # stellar masses at each redshift defined in the sampler.mass_grid...if the grid is the same for every redshift,
    # then this doesn't matter and the below is correct. Ignoring for now.
    N_sf = integrate(sf_sampler, cosmo, mmin_sf, mmax_sf, zmin_sf, zmax_sf) * f_sky(area_deg2)
    N_q = integrate(q_sampler, cosmo, mmin_q, mmax_q, zmin_q, zmax_q) * f_sky(area_deg2)

    # Poisson sample, if requested
    N_sf = poisson ? rand(rng, Poisson(N_sf)) : round(Int, N_sf)
    N_q = poisson ? rand(rng, Poisson(N_q)) : round(Int, N_q)

    @info "Number of star-forming galaxies: $N_sf"
    @info "Number of quiescent galaxies: $N_q"

    # TODO: Even if N_sf and N_q are very large, break them up into chunks of some smaller number of galaxies,
    # sample the stellar masses and redshifts, and then cull the lists using `get_mass_limit` to avoid memory issues.
    N_sf > 1e7 && @warn "The expected number of star-forming galaxies ($N_sf) is very large; this may take a long time and use a lot of memory or crash outright."
    N_q > 1e7 && @warn "The expected number of quiescent galaxies ($N_q) is very large; this may take a long time and use a lot of memory or crash outright."

    # Sample redshifts and stellar masses
    sf = rand(rng, sf_sampler, N_sf) # 2 x N_sf array of (Mstar, z)
    q = rand(rng, q_sampler, N_q)    # 2 x N_q array of (Mstar, z)

    rng = use_rng ? rng : nothing

    # Get first result so we know the output return type
    r1 = egg(sf[1,1], sf[2,1], true, filters, zpts; rng, kws...)
    results = Vector{typeof(r1)}(undef, N_sf + N_q)
    results[1] = r1
    Threads.@threads for i in 1:N_sf
        results[i] = egg(sf[1, i], sf[2, i], true, filters, zpts; rng, cosmo, kws...)
    end
    Threads.@threads for i in 1:N_q
        results[N_sf + i] = egg(q[1, i], q[2, i], false, filters, zpts; rng, cosmo, kws...)
    end

    return results
end

end # module