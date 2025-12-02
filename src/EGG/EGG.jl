# EGG uses separate double Schechter mass functions for the star-forming galaxy (SFG) population and the quiescent galaxy (QG) population, so we separate implementation based on whether the *galaxy* as a whole is quiescent or star-forming. Within these classifications, galaxies can also vary in whether their bulges are SF or not

"""
Contains code to generate galaxy catalogs using methods similar to those used in the Empirical Galaxy Generator (Schreiber2017)
"""
module EGG

using ArgCheck: @argcheck, @check
using Distributions: LogNormal, Normal, Uniform
using FITSIO: FITS
using IrrationalConstants: logten
using Random: Random, default_rng, AbstractRNG
using SpecialFunctions: erf

export egg

include("optlib.jl")

"""
    uv_vj(logMstar, z, SF::Bool)
Takes `log10(Mstar [M⊙])`, redshift, and `SF::Bool` determining whether the stellar population is star-forming (`true`) or quiescent (`false`).

Returns U-V and V-J colors `(uv, vj)`.
"""
function uv_vj(logMstar, z, SF::Bool; rng::AbstractRNG=default_rng())
    return if SF
        a0 = 0.48 * erf(logMstar - 10) + 1.15
        a1 = -0.28 + 0.25 * max(0, logMstar - 10.35)
        vj = a0 + a1 * min(z, 3.3) # V-J color for star-forming galaxies
        vj = min(vj, 1.7) # limit to <1.7
        vj = rand(rng, Normal(vj, 0.1)) # Add first error
        uv = 0.65 * vj + 0.45
        vj = rand(rng, Normal(vj, 0.12)) # Add extra error
        uv = rand(rng, Normal(uv, 0.12)) # Add extra error
        (uv, vj)
    else
        vj = 0.1 * (logMstar - 11) + 1.25
        vj = rand(rng, Normal(vj, 0.1)) # Add first error
        vj = max(min(vj, 1.45), 1.15) # Restrict 1.15 <= V-J <= 1.45
        uv = 0.88 * vj + 0.75
        vj = rand(rng, Normal(vj, 0.1)) # Add extra error
        uv = rand(rng, Normal(uv, 0.1)) # Add extra error
        (uv, vj)
    end
end

# SF is whether galaxy is star-forming or not
function egg(Mstar, z, SF::Bool; rng::AbstractRNG=default_rng()) 
    logMstar = log10(Mstar)
    log1pz = log1p(z) / logten # same as log10(z + 1)
    # Distributions for the bulge-to-total mass ratio
    BT = if SF
        rand(rng, LogNormal(log(exp10(-0.7 + 0.27 * (logMstar - 10))), 0.2 * logten))
    else
        rand(rng, LogNormal(log(exp10(-0.3 + 0.1 * (logMstar - 10))), 0.2 * logten))
    end
    Mbulge = Mstar * BT
    Mdisk = Mstar - Mbulge

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
    R50_disk = rand(rng, LogNormal(log(exp10(0.2 * (logMstar - 9.35) + Fz_disk)), 0.17 * logten)) # kpc
    R50_bulge = rand(rng, LogNormal(log(exp10(0.2 * (logMstar - 11.25) + Fz_bulge)), 0.2 * logten)) # kpc
    α_R = 1 - 0.8 * log10(R50_disk / R50_bulge)
    BT_α = BT^α_R
    R50_tot = R50_disk * (1 - BT_α) + R50_bulge * BT_α
    PA = rand(rng, Uniform(0, 360)) # Uniform position angle, shared between bulge and disk

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

    # Colors
    uv_disk, vj_disk = uv_vj(logMstar, z, disk_SF)
    uv_bulge, vj_bulge = uv_vj(logMstar, z, bulge_SF)

    return (Mstar = Mstar, uv_disk = uv_disk, vj_disk = vj_disk, uv_bulge = uv_bulge, vj_bulge = vj_bulge, R50_disk = R50_disk, R50_bulge = R50_bulge, R50 = R50_tot, PA = PA, BT = BT, Mdisk = Mdisk, Mbulge = Mbulge)
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