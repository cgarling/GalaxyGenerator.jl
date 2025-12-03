abstract type IGMAttenuation end
Base.Broadcast.broadcastable(m::IGMAttenuation) = Ref(m)

# Generic methods
"""
    transmission(model::IGMAttenuation, z, λ_r)
Returns the IGM transmission for light emitted at rest wavelength `λ_r` (in microns) at redshift `z` given the provided IGM transmission model `model`. Defined as `exp(-τ)` where `τ` is the optical depth. Generic implementation is `transmission(tau(model, z, λ_r))`.
"""
transmission(model::IGMAttenuation, z, λ_r) = transmission(tau(model, z, λ_r))

"""
    transmission(tau) = exp(-tau)
Convert optical depth (`tau`) to transmission (`exp(-tau)`).
"""
transmission(tau) = exp(-tau)

#############
# NoIGM model
"""
`transmission(::NoIGM, z, λ_r)` always returns `one(z)`, adding no attenuation.
"""
struct NoIGM <: IGMAttenuation end

"""
    tau(model::IGMAttenuation, z, λ_r)
Returns the IGM optical depth for light emitted at rest wavelength `λ_r` (in microns) at redshift `z` given the provided IGM transmission model `model`.
"""
function tau end
tau(::NoIGM, z, λ_r) = zero(z)
transmission(::NoIGM, z, λ_r) = one(z) # Define this directly to avoid the generic exp(-tau) method

#################
# Madau1995 model 
"""
`transmission(::Madau1995IGM, z, λ_r)` returns the Madau+95 IGM transmission for wavlength bands for rest-frame wavelength `λ_r` (in microns) emitted at redshift `z`.
"""
struct Madau1995IGM <: IGMAttenuation end

"""Madau+95 Lyman-series wavelengths in Angstroms."""
const madau_lyw = (1215.67, 1025.72, 972.537, 949.743, 937.803,
        930.748, 926.226, 923.150, 920.963, 919.352,
        918.129, 917.181, 916.429, 915.824, 915.329,
        914.919, 914.576)

"""Madau+95 Lyman-series coefficients."""
const madau_lycoeff = (0.0036, 0.0017, 0.0011846, 0.0009410, 0.0007960,
        0.0006967, 0.0006236, 0.0005665, 0.0005200, 0.0004817,
        0.0004487, 0.0004200, 0.0003947, 0.0003720, 0.0003520,
        0.0003334, 0.00031644)

function tau(::Madau1995IGM, z, λ_r)
    λ_r *= 1e4 # Convert from input microns to Angstroms
    lylim   = 911.75
    a_metal = 0.0017

    z1   = 1 + z
    λ_o  = λ_r * z1
    xc   = λ_o / lylim
    if z1 > xc / 0.8
        @warn "Madau1995IGM model has poor performance at short wavelengths. As a precaution, this function will return the transmission value evaluated at λ_rest = 729.4 Å." maxlog=1
        return transmission(Madau1995IGM(), z, lylim * 0.8) # z1 = xc = λ_r * z1 / lylim, solve for λ_r
    end
    # Lyman series line blanketing
    tau = 0.0

    for i in eachindex(madau_lyw)
        if λ_r > madau_lyw[i]
            continue
        end
        tau += madau_lycoeff[i] * (λ_o / madau_lyw[i])^3.46

        if i == 1   # Lyα: add metal blanketing
            tau += a_metal * (λ_o / madau_lyw[i])^1.68
        end
    end

    # LyC absorption (λ < 912 Å)
    if λ_r < lylim
        tau +=
            0.25 * xc^3 * (z1^0.46 - xc^0.46) +
            9.4  * xc^1.5 * (z1^0.18 - xc^0.18) +
            0.7  * xc^3   * (z1^(-1.32) - xc^(-1.32)) -
            0.023 * (z1^1.68 - xc^1.68)
    end
    # Convert τ to transmission and return
    return max(0.0, tau)
end


struct Inoue2014IGM <: IGMAttenuation end