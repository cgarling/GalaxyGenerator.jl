# EGG uses separate double Schechter mass functions for the star-forming galaxy (SFG) population and the quiescent galaxy (QG) population, so we separate implementation based on whether the *galaxy* as a whole is quiescent or star-forming. Within these classifications, galaxies can also vary in whether their bulges are SF or not

"""
Contains code to generate galaxy catalogs using methods similar to those used in the Empirical Galaxy Generator (Schreiber2017)
"""
module EGG

using ..GalaxyGenerator: interp_lin

using ArgCheck: @argcheck, @check
using DataInterpolations: LinearInterpolation
using Distributions: LogNormal, Normal, Uniform
using FITSIO: FITS
using IrrationalConstants: logten
using PhotometricFilters: AbstractFilter, magnitude
using Random: Random, default_rng, AbstractRNG
using SpecialFunctions: erf
using StaticArrays: SVector
using Statistics: mean

export egg

include("optlib.jl")
# Load default optical SED library
const optlib = OptLib(joinpath(@__DIR__, "data", "opt_lib_fast.fits"))

include("IGM.jl") # IGM attenuation models

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
function uv_vj(logMstar, z, SF::Bool, ; rng::AbstractRNG=default_rng())
    # This implementation is based on the code in the egg repository
    uv, vj = if SF
        a0 = 0.58 * erf(logMstar - 10) + 1.39
        a1 = -0.34 + 0.3 * max(0, logMstar - 10.35)
        vj = a0 + a1 * min(z, 3.3)
        vj = min(vj, 2.0)
        rnd_amp = 0.2 + (0.25 + 0.12 * clamp((z - 0.5) / 2.0, 0.0, 1.0)) *
                max(1.0 - 2.0*abs(logMstar - (10.3 + 0.4 * erf(z - 1.5))), 0.0)
        vj = rand(rng, Normal(vj, rnd_amp))

        # Move in UVJ diagram according to UVJ vector
        slope = 0.65
        theta = atan(slope) # This can be simplified but don't care right now
        vj = 0.0 + vj * cos(theta)
        uv = 0.45 + vj * sin(theta)
        vj = rand(rng, Normal(vj, 0.15))
        uv = rand(rng, Normal(uv, 0.15))
        (uv, vj)
    else
        vj = 0.1 * (logMstar - 11) + 1.25
        vj = rand(rng, Normal(vj, 0.1)) # Add first error
        vj = max(min(vj, 1.45), 1.15) # Restrict 1.15 <= V-J <= 1.45
        uv = 0.88 * vj + 0.6
        vj = rand(rng, Normal(vj, 0.1)) # Add extra error
        uv = rand(rng, Normal(uv, 0.1)) # Add extra error
        (uv, vj)
    end
    # Add additional color offset depending on redshift
    uv += 0.4 * max((0.5 - z) / 0.5, 0.0)
    vj += 0.2 * max((0.5 - z) / 0.5, 0.0)
    return uv, vj
end

# SF is whether galaxy is star-forming or not
function egg(Mstar, z, SF::Bool, @nospecialize(filters::AbstractVector{<:AbstractFilter});
    rng::AbstractRNG=default_rng(), 
    optlib::OptLib=optlib,
    igm::IGMAttenuation=Inoue2014IGM())

    logMstar = log10(Mstar)
    log1pz = log1p(z) / logten # same as log10(z + 1)
    # Distributions for the bulge-to-total mass ratio
    BT = if SF
        rand(rng, LogNormal(log(exp10(-0.7 + 0.27 * (logMstar - 10))), 0.2 * logten))
    else
        rand(rng, LogNormal(log(exp10(-0.3 + 0.1 * (logMstar - 10))), 0.2 * logten))
    end
    BT = clamp(BT, 0.0, 1.0)
    Mbulge = Mstar * BT
    Mdisk = Mstar - Mbulge
    @check Mbulge >=0
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
    R50_disk = exp10(0.2 * (min(10.6, logMstar) - 9.35) + Fz_disk)
    R50_disk = rand(rng, LogNormal(log(R50_disk), 0.17 * logten)) # kpc
    R50_bulge = rand(rng, LogNormal(log(exp10(0.56 * (logMstar - 11.25) + Fz_bulge)), 0.2 * logten)) # kpc
    α_R = 1 - 0.8 * log10(R50_disk / R50_bulge)
    BT_α = BT^α_R
    R50_tot = R50_disk * (1 - BT_α) + R50_bulge * BT_α
    PA = rand(rng, Uniform(-180, 180)) # Uniform position angle, shared between bulge and disk

    disk_SF = SF # Disks are SF if galaxy is SF
    bulge_SF = if ~SF # Bulge is quiescent if galaxy is quiescent
        false
    else
        if BT >= 0.6 # Bulge is quiescient if galaxy is bulge-dominated
            false
        else
            rand(rng, (true, false)) # Otherwise, 50% chance of quiescent/SF
        end
    end

    # SFRs, initially in log10(sfr) unit
    ms_disp = 0.3 # Dispersion of SFMS in dex; EGG notes this can change quality of simulation
    sfrms = logMstar - 9.67 + 1.82 * log1pz - 0.38 * max(0.0, logMstar - 9.59 - 2.22 * log1pz)^2
    sfr = if SF 
        rand(rng, Normal(sfrms, ms_disp))
    else # If passive,
        rand(rng, Normal(min(sfrms, 0.5 * (logMstar - 11) + log1pz - 0.6), 0.4))
    end
    # Add starburst, 3.3% of star-forming galaxies
    if SF && rand(rng) < 0.033
        sfr += 0.72 # Enhance SFR by a factor of 5.24
    end
    rsb = sfr - sfrms # Ratio of SFR / SFR_MS
    sfr = exp10(sfr) # Convert to linear units
    # IRX = log10(L_IR / L_UV)
    irx = (0.45 * min(z, 3.0) + 0.35) * (logMstar - 10.5) + 1.2
    irx = exp10(rand(rng, Normal(irx, 0.4)))
    sfr_ir = sfr / (1 + 1 / irx)
    sfr_uv = sfr / (1 + irx)

    # Optical colors
    uv_disk, vj_disk = uv_vj(logMstar, z, disk_SF)
    uv_bulge, vj_bulge = uv_vj(logMstar, z, bulge_SF)

    # Get optical SEDs; SEDs returned in units of L⊙ / μm / M⊙
    m2l_cor = get_m2l_cor(z) # M/L correction in dex
    opt_λ_disk, opt_sed_disk = get_opt_sed(uv_disk, vj_disk, optlib)
    opt_λ_bulge, opt_sed_bulge = get_opt_sed(uv_bulge, vj_bulge, optlib)
    # Convert SED to units of erg Å^-1 cm^-2 s^-1
    # 1 * UnitfulAstro.Lsun / u"μm" / (4π * (10u"pc")^2) |> u"erg" / u"s" / u"cm^2" / u"angstrom"
    opt_sed_disk .*= exp10(log10(Mdisk) - m2l_cor) * 3.1993443f-11
    opt_sed_bulge .*= exp10(log10(Mbulge) - m2l_cor) * 3.1993443f-11
    # Redshift the wavelengths
    opt_λ_disk .*= 1 + z
    opt_λ_bulge .*= 1 + z

    # IR properties
    tdust_ms = 32.13 + 4.6 * (z - 2) # T_dust main sequence value at z
    tdust = tdust_ms + 10.1 * rsb # Starbursts are warmer
    # Massive galaxies are colder (=downfall of SFE)
    tdust -= 1.5 * max(0.0, 2 - z) * clamp(logMstar - 10.7, 0.0, 1.0)
    ir8 = (4.08 + 3.29 * clamp(z - 1, 0.0, 1.0)) * 0.81
    ir8 *= exp10(0.66 * max(0.0, rsb)) # Starbursts have higher ir8
    ir8 *= exp10(-clamp(logMstar - 10, -1.0, 0.0)) # low-mass galaxies have larger ir8
    tdust = rand(rng, Normal(tdust, 0.12 * tdust_ms))
    ir8 *= exp10(0.18 * randn(rng)) # Ratio of IR to 8 μm luminosity (Elbaz 2011)
    lir = sfr_ir / 1.72e-10 # Infrared luminosity from 8 to 1000 μm in L⊙
    ir8 = clamp(ir8, 0.48, 27.5) # range allowed in IR library
    fpah = clamp(1 / (1 - (331 - 691 * ir8) / (193 - 6.98 * ir8)), 0.0, 1.0)
    if !isfinite(lir)
        lir = 0.0
    end

    # Get IR SED
    # Merge optical and IR SEDs
    # Generate emission lines
    # Add emission lines
    # Obtain rest-frame magnitudes
    # Apply IGM absorption, MW dust absorption
    # Redshift SED, obtain observed magnitudes

    return (Mstar = Mstar, sfr=sfr, sfr_ir = sfr_ir, sfr_uv = sfr_uv, uv_disk = uv_disk, vj_disk = vj_disk, uv_bulge = uv_bulge, vj_bulge = vj_bulge, R50_disk = R50_disk, R50_bulge = R50_bulge, R50 = R50_tot, PA = PA, BT = BT, Mdisk = Mdisk, Mbulge = Mbulge, Tdust = tdust, fpah = fpah, lir = lir, ir8 = ir8, IRX = irx)
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