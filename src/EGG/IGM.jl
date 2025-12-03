abstract type IGMAttenuation end
Base.Broadcast.broadcastable(m::IGMAttenuation) = Ref(m)

# struct ConstantIGM end

struct NoIGM <: IGMAttenuation end
transmission(::NoIGM, z) = one(z)
"""
`transmission(::Madau1995IGM, z)` returns a 3–element vector with Madau+95 IGM transmission efficiency for wavlength bands `[ <912Å band>, <920–1015Å band>, <1050–1170Å band> ]`.
"""
struct Madau1995IGM <: IGMAttenuation end
function transmission(::Madau1995IGM, z, λ_r)
    # Ly-series wavelengths and coefficients (Madau 1995)
    lyw = (1215.67, 1025.72, 972.537, 949.743, 937.803,
        930.748, 926.226, 923.150, 920.963, 919.352,
        918.129, 917.181, 916.429, 915.824, 915.329,
        914.919, 914.576)

    lycoeff = (0.0036, 0.0017, 0.0011846, 0.0009410, 0.0007960,
        0.0006967, 0.0006236, 0.0005665, 0.0005200, 0.0004817,
        0.0004487, 0.0004200, 0.0003947, 0.0003720, 0.0003520,
        0.0003334, 0.00031644)

    lylim   = 911.75
    a_metal = 0.0017

    z1   = 1 + z
    λ_o  = λ_r * z1
    xc   = λ_o / lylim
    if z1 > xc / 0.8
        @warn "Madau1995IGM model has poor performance at short wavelengths. As a precaution, this function will return the transmission value evaluated at λ_rest = 729.4 Å." maxlog=1
        return transmission(Madau1995IGM(), z, lylim * 0.8) # z1 = xc = λ_r * z1 / lylim, solve for λ_r
    end
    # ----------------------------------------------------------
    # Ly-series line blanketing
    # ----------------------------------------------------------
    tau = 0.0

    for i in eachindex(lyw)
        if λ_r > lyw[i]
            continue
        end
        tau += lycoeff[i] * (λ_o / lyw[i])^3.46

        if i == 1   # Lyα: add metal blanketing
            tau += a_metal * (λ_o / lyw[i])^1.68
        end
    end

    # ----------------------------------------------------------
    # LyC absorption (λ < 912 Å)
    # ----------------------------------------------------------
    if λ_r < lylim
        tau +=
            0.25 * xc^3 * (z1^0.46 - xc^0.46) +
            9.4  * xc^1.5 * (z1^0.18 - xc^0.18) +
            0.7  * xc^3   * (z1^(-1.32) - xc^(-1.32)) -
            0.023 * (z1^1.68 - xc^1.68)
    end
    # Convert τ to transmission and return
    return min(1.0, exp(-tau))
end


struct Inoue2014IGM <: IGMAttenuation end