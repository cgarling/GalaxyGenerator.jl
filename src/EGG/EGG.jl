# EGG uses separate double Schechter mass functions for the star-forming galaxy (SFG) population and the quiescent galaxy (QG) population, so we separate implementation based on whether the *galaxy* as a whole is quiescent or star-forming. Within these classifications, galaxies can also vary in whether their bulges are SF or not

"""
Contains code to generate galaxy catalogs using methods similar to those used in the Empirical Galaxy Generator (Schreiber2017)
"""
module EGG

using ..GalaxyGenerator: interp_lin, interp_log, merge_add
using ..GalaxyGenerator.IGM: IGMAttenuation, transmission, tau, Inoue2014IGM
using ..GalaxyGenerator.MassFunctions: BinnedRedshiftMassFunction, DoubleSchechterMassFunction, integrate

using ArgCheck: @argcheck, @check
using Cosmology: AbstractCosmology, distmod, Planck18
using DataInterpolations: LinearInterpolation
using Distributions: LogNormal, Normal, Uniform
using FITSIO: FITS
using IrrationalConstants: logten
using PhotometricFilters: AbstractFilter, magnitude, MagnitudeSystem, mean_flux_density, zeropoint_mag, detector_type, wavelength, throughput
using Random: Random, default_rng, AbstractRNG
using SpecialFunctions: erf
using StaticArrays: SVector
using Statistics: mean
import Unitful as u

export egg

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
                   igm::IGMAttenuation=Inoue2014IGM())

Given a redshift `z`, whether the galaxy is star-forming (`SF::Bool`), an apparent magnitude limit `mag_lim` in the filter `filt` with magnitude system `mag_sys`, returns the stellar mass limit `Mstar` (in M⊙) required to reach that magnitude limit.

Below we calculate the stellar mass limit at redshift 2 for star-forming galaxies with an apparent magnitude limit of 25 in the HST/ACS F606W filter in the Vega magnitude system.

```jldoctest
julia> using GalaxyGenerator.EGG: get_mass_limit

julia> using PhotometricFilters: HST_ACS_WFC_F606W, Vega

julia> result = get_mass_limit(2.0, true, 25.0, HST_ACS_WFC_F606W(), Vega());

julia> isapprox(result, 1.1456e12; rtol=1e-2)
true
```
"""
function get_mass_limit(z, SF::Bool, mag_lim, filt::AbstractFilter, mag_sys::MagnitudeSystem; 
    Mstar=logrange(1e6, 1e13; length=100), cosmo::AbstractCosmology=Planck18, optlib::OptLib=optlib,
    igm::IGMAttenuation=Inoue2014IGM())
    # Generate Mstar - flux relation
    m2l_cor = get_m2l_cor(z) # M/L correction in dex
    zpt = zeropoint_mag(filt, mag_sys)

    mags = Vector{Float64}(undef, length(Mstar))
    for i in eachindex(Mstar)
        # Get UV, VJ colors, optical SED
        uv, vj = uv_vj(log10(Mstar[i]), z, SF; rng=nothing)
        λ, sed, Av = get_opt_sed(uv, vj, optlib)
        # Convert SED to units of erg Å^-1 cm^-2 s^-1, place at 10 pc for absolute mags
        sed .*= exp10(log10(Mstar[i]) - m2l_cor) * 3.1993443f-11
        # Add IGM attenuation; this takes most of the runtime
        sed .*= transmission.(igm, z, λ)
        λ .*= 1 + z # Redshift wavelengths
        λ .*= 1f4   # convert μm to Å
        # mags[i] = magnitude(filt, mag_sys, λ * u.μm, sed * u.erg / u.s / u.cm^2 / u.angstrom) + distmod(cosmo, z)
        fbar = mean_flux_density(λ, sed, filt.(λ), detector_type(filt))
        mags[i] = magnitude(fbar, zpt) + distmod(cosmo, z)
    end
    if mag_lim > first(mags)
        @warn "The minimum stellar mass $(first(Mstar)) provided to `get_mass_limit` for redshift $z is too high to sample galaxies to the requested magnitude limit $mag_lim. Returning $(first(Mstar))."
        return first(Mstar)
    elseif mag_lim < last(mags)
        @warn "The maximum stellar mass $(last(Mstar)) provided to `get_mass_limit` for redshift $z is too low to sample galaxies brighter than the requested magnitude limit $mag_lim. Returning $(last(Mstar))."
        return last(Mstar)
    else
        # return Mstar[searchsortedfirst(mags, mag_lim; rev=true)]
        mass_lim = interp_log(reverse(mags), reverse(Mstar), mag_lim)
        mass_lim /= 2 # Add safety factor of 2
        return max(first(Mstar), mass_lim) # Don't go below minimum Mstar provided
    end
end

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
        vj = a0 + a1 * min(z, 3.3)
        vj = min(vj, 2.0)
        rnd_amp = 0.2 + (0.25 + 0.12 * clamp((z - 0.5) / 2.0, 0.0, 1.0)) *
                max(1.0 - 2.0*abs(logMstar - (10.3 + 0.4 * erf(z - 1.5))), 0.0)
        vj = isnothing(rng) ? vj : rand(rng, Normal(vj, rnd_amp))

        # Move in UVJ diagram according to UVJ vector
        slope = 0.65
        theta = atan(slope) # This can be simplified but don't care right now
        vj = 0.0 + vj * cos(theta)
        uv = 0.45 + vj * sin(theta)
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

    Mstar, z = Float32(Mstar), Float32(z)
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
    α_R = 1 - 0.8 * log10(R50_disk / R50_bulge)
    BT_α = BT^α_R
    R50_tot = R50_disk * (1 - BT_α) + R50_bulge * BT_α
    # Uniform position angle, shared between bulge and disk
    PA = isnothing(rng) ? 0.0 : rand(rng, Uniform(-180, 180))

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

    return (Mstar = Mstar, sfr = sfr, sfr_ir = sfr_ir, sfr_uv = sfr_uv, uv_disk = uv_disk, vj_disk = vj_disk, uv_bulge = uv_bulge, vj_bulge = vj_bulge, R50_disk = R50_disk, R50_bulge = R50_bulge, R50 = R50_tot, PA = PA, BT = BT, Mdisk = Mdisk, Mbulge = Mbulge, Tdust = tdust, fpah = fpah, lir = lir, ir8 = ir8, IRX = irx, OH = oh)
end

# This method takes 
function egg(Mstar, z, SF::Bool, 
    @nospecialize(filters::AbstractVector{<:AbstractFilter}),
    @nospecialize(mag_sys::AbstractVector{<:MagnitudeSystem});
    cosmo::AbstractCosmology=Planck18,
    rng::Union{Nothing,AbstractRNG}=default_rng(), 
    optlib::OptLib=optlib,
    irlib::IRLib=irlib,
    igm::IGMAttenuation=Inoue2014IGM())

    @argcheck length(filters) == length(mag_sys)
    Mstar, z = Float32(Mstar), Float32(z)
    logMstar = log10(Mstar)
    dmod = distmod(cosmo, z)

    # Call above method to calculate galaxy properties
    r = egg(Mstar, z, SF; rng = rng)

    # Get optical SEDs; SEDs returned in units of L⊙ / μm / M⊙, λ in μm
    m2l_cor = get_m2l_cor(z) # M/L correction in dex
    λ_disk, sed_disk, Av_disk = get_opt_sed(r.uv_disk, r.vj_disk, optlib)
    λ_bulge, sed_bulge, Av_bulge = get_opt_sed(r.uv_bulge, r.vj_bulge, optlib)
    # Convert SED to units of erg Å^-1 cm^-2 s^-1 and place at distance of 10 pc for absolute mags
    # 1 * UnitfulAstro.Lsun / u"μm" / (4π * (10u"pc")^2) |> u"erg" / u"s" / u"cm^2" / u"angstrom"
    sed_disk .*= exp10(log10(r.Mdisk) - m2l_cor) * 3.1993443f-11
    sed_bulge .*= exp10(log10(r.Mbulge) - m2l_cor) * 3.1993443f-11

    # Get IR SED
    ir_result = get_ir_sed(r.Tdust, irlib)
    ir_λ = ir_result.lam # microns

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

    # Calculate final ir_sed in units of erg Å^-1 cm^-2 s^-1
    ir_sed = if typeof(irlib) == CS17_IRLib
        # CS_17 library is in units of L⊙ / μm / M⊙ of dust
        (ir_result.dust .* (1 - fpah) .+ ir_result.pah .* fpah) .* Mdust .* 3.1993443f-11
    else
        ir_result.sed .* 3.1993443f-11 # Assume in L⊙ / μm
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
    # return λ, sed

    # Obtain rest-frame magnitudes
    # This is inefficient as `magnitude` resamples the filter to the λ every time it's called; don't care for now
    mag_abs = [magnitude(filters[i], mag_sys[i], λ * u.μm, sed * u.erg / u.s / u.cm^2 / u.angstrom) for i in eachindex(filters)]

    # Apply IGM absorption, MW dust absorption
    tau_igm = tau.(igm, z, λ) # This is working, but expensive ~ 1 ms for full SED
    A_λ = 0.0 # Get MW extinction
    tau_MW = A_λ / (2.5 * log10(ℯ)) # The constant is ~1.086
    τ = tau_igm # in place
    τ .= tau_igm .+ tau_MW
    sed .*= exp.(-τ)
    # In EGG, IGM absorption is only applied in 3 wavelength bins, I think
    # we will do full transmission over the entire SED. For this reason EGG
    # has to compute IGM transmission separately for the lines and for the binned SED,
    # but we will just add the emission lines to the SED and then apply the IGM absorption once

    # Redshift SED, obtain observed magnitudes
    λ .*= 1 + z
    # This is inefficient as `magnitude` resamples the filter to the λ every time it's called; don't care for now
    mag_obs = [magnitude(filters[i], mag_sys[i], λ * u.μm, sed * u.erg / u.s / u.cm^2 / u.angstrom) + dmod for i in eachindex(filters)]

    return merge(r, (Mgas = Mgas, MH2 = MH2, fpah = fpah, Mdust = Mdust, mag_abs = mag_abs, mag_obs = mag_obs, λ = λ, sed = sed))
    # mag_obs = [-25//10 * log10(mean_flux_density(filters[i], λ, sed)) - zeropoint_mag(filters[i], mag_sys[i]) for i in eachindex(filters)]
    # function _itp(f::AbstractFilter, λ)
    #     wave = wavelength(f)
    #     λv = @view λ[searchsortedfirst(λ, first(wave)):searchsortedlast(λ, last(wave))]

    # mag_obs = [-25//10 * log10(mean_flux_density(λ, sed, [interp_lin(wavelength(filters[i]), throughput(filters[i]), j * u.angstrom) for j in λ], detector_type(filters[i]))) - zeropoint_mag(filters[i], mag_sys[i]) for i in eachindex(filters)]

end

# function EGG(Mstar, z; rng::AbstractRNG=default_rng())
#     logMstar = log10(Mstar)
#     log1pz = log1p(z) / logten # same as log10(z + 1)
#     # Distributions for the bulge-to-total mass ratio
#     bt_dist_SF = LogNormal(log(exp10(-0.7 + 0.27 * (logMstar - 10))), 0.2 * logten)
#     bt_dist_Q = LogNormal(log(exp10(-0.3 + 0.1 * (logMstar - 10))), 0.2 * logten)

#     # Distributions for bulge, disk sizes
#     Fz_disk = if z <= 1.7
#         0.41 - 0.22 * log1pz
#     else
#         0.62 - 0.7 * log1pz
#     end
#     Fz_bulge = if z <= 0.5
#         0.78 - 0.6 * log1pz
#     else
#         0.9 - 1.3 * log1pz
#     end
#     R50_disk = LogNormal(log(exp10(0.2 * (logMstar - 9.35) + Fz_disk)), 0.17 * logten) # kpc
#     R50_bulge = LogNormal(log(exp10(0.2 * (logMstar - 11.25) + Fz_bulge)), 0.2 * logten) # kpc
#     PA_dist = Uniform() # Uniform position angle, shared between bulge and disk

#     # Colors
#     # For star-forming case
#     a0 = 0.48 * erf(logMstar - 10) + 1.15
#     a1 = -0.28 + 0.25 * max(0, logMstar - 10.35)
#     vj_sf = a0 + a1 * min(z, 3.3) # V-J color for star-forming galaxies
#     vj_sf = min(vj_sf, 1.7) # limit to <1.7
#     # vj_sf = rand(Normal(vj_sf, 0.1)) # Add first error
#     uv_sf = 0.65 * vj_sf + 0.45
#     # vj_sf = rand(Normal(vj_sf, 0.12)) # Add extra error
#     # uv_sf = rand(Normal(uv_sf, 0.12)) # Add extra error
#     # For quiescent case
#     vj_q = 0.1 * (logMstar - 11) + 1.25
#     # vj_q = rand(Normal(vj_q, 0.1)) # Add first error
#     vj_q = max(min(vj_q, 1.45), 1.15) # Restrict 1.15 <= V-J <= 1.45
#     uv_q = 0.88 * vj_q + 0.75
#     # vj_q = rand(Normal(vj_q, 0.1)) # Add extra error
#     # uv_q = rand(Normal(uv_q, 0.1)) # Add extra error

#     # Proceed assuming all disks are "star-forming"
#     # bulges can be star-forming or quiescent with equal probability
#     bulge_sf = rand((true,false))
# end

end # module