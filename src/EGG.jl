# Contains code to generate galaxy catalogs using methods similar to those used in the Empirical Galaxy Generator (Schreiber2017)


function EGG(Mstar, z; rng::AbstractRNG=default_rng())
    logMstar = log10(Mstar)
    log1pz = log1p(z) / logten # same as log10(z + 1)
    # Distributions for the bulge-to-total mass ratio
    bt_dist_SF = LogNormal(log(exp10(-0.7 + 0.27 * (logMstar - 10))), 0.2 * logten)
    bt_dist_Q = LogNormal(log(exp10(-0.3 + 0.1 * (logMstar - 10))), 0.2 * logten)

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
    R50_disk = LogNormal(log(exp10(0.2 * (logMstar - 9.35) + Fz_disk)), 0.17 * logten) # kpc
    R50_bulge = LogNormal(log(exp10(0.2 * (logMstar - 11.25) + Fz_bulge)), 0.2 * logten) # kpc
    PA_dist = Uniform() # Uniform position angle, shared between bulge and disk

    # Colors
    # For star-forming case
    a0 = 0.48 * erf(logMstar - 10) + 1.15
    a1 = -0.28 + 0.25 * max(0, logMstar - 10.35)
    vj_sf = a0 + a1 * min(z, 3.3) # V-J color for star-forming galaxies
    vj_sf = min(vj_sf, 1.7) # limit to <1.7
    # vj_sf = rand(Normal(vj_sf, 0.1)) # Add first error
    uv_sf = 0.65 * vj_sf + 0.45
    # vj_sf = rand(Normal(vj_sf, 0.12)) # Add extra error
    # uv_sf = rand(Normal(uv_sf, 0.12)) # Add extra error
    # For quiescent case
    vj_q = 0.1 * (logMstar - 11) + 1.25
    # vj_q = rand(Normal(vj_q, 0.1)) # Add first error
    vj_q = max(min(vj_q, 1.45), 1.15) # Restrict 1.15 <= V-J <= 1.45
    uv_q = 0.88 * vj_q + 0.75
    # vj_q = rand(Normal(vj_q, 0.1)) # Add extra error
    # uv_q = rand(Normal(uv_q, 0.1)) # Add extra error

    # Proceed assuming all disks are "star-forming"
    # bulges can be star-forming or quiescent with equal probability
    bulge_sf = rand((true,false))
end