abstract type IGMAttenuation end
Base.Broadcast.broadcastable(m::IGMAttenuation) = Ref(m)

# struct ConstantIGM end

struct NoIGM <: IGMAttenuation end
transmission(::NoIGM, z) = one(z)
"""
`transmission(::Madau1995IGM, z)` returns a 3–element vector with Madau+95 IGM transmission efficiency for wavlength bands `[ <912Å band>, <920–1015Å band>, <1050–1170Å band> ]`.
"""
struct Madau1995IGM <: IGMAttenuation end
    # ptau2::T
    # ptau3::T
# This implementation is based on the one in EGG, but it doesn't seem to replicate
# Madau Figure 3 as well. GOing to try FSPS implementation
# function transmission(::Madau1995IGM, z, λ_r) # λ in microns
#     # T = promote_type(typeof(z), typeof(λ))
#     λ_α = 0.1216 # Lyman α
#     λ_β = 0.1026 # Lyman β
#     λ_γ = 0.0973 # Lyman γ
#     λ_δ = 0.095  # Lyman δ
#     λ_L = 0.0912 # Lyman limit
#     if λ_r < λ_L
#         return 0.0 # No transmission shorter than Lyman limit
#     end
#     λ_o = λ_r * (1 + z)
#     # Equation 12, optical depth due to Lyman-α forest
#     # L_α_τ = if λ_β <= λ_r <= λ_α 
#     #     0.0036 * (λ_o / λ_α)^3.46
#     # else
#     #     0.0
#     # end
#     # Equation 15, optical depth due to higher order Lyman series lines

#     L_series_τ = if λ_r > 0.1170
#         0.0
#     elseif λ_L <= λ_r <= 0.1015
#         1.7e-3*(λ_o/λ_β)^3.46 +
#         1.2e-3*(λ_o/λ_γ)^3.46 +
#         9.3e-4*(λ_o/λ_δ)^3.46
#     else
#         3.6e-3*(λ_o/λ_α)^3.46
#     end
#     # return exp(-(L_α_τ + L_series_τ))
#     return exp(-L_series_τ)
# end
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
# function transmission(::Madau1995IGM, z::T) where T
#     # --- Band 1: <912Å (no flux)
#     out1= 0

#     # --- Band 2: 920–1015 Å (rest)
#     l0 = 920*(1+z)
#     l1 = 1015*(1+z)
#     tl = range(l0, l1; length=100)
#     ptau = @. exp(-1.7e-3*(tl/1026)^3.46 -
#                   1.2e-3*(tl/972.5)^3.46 -
#                   9.3e-4*(tl/950)^3.46)
#     out2 = mean(ptau)

#     # --- Band 3: 1050–1170 Å (rest)
#     l0 = 1050*(1+z)
#     l1 = 1170*(1+z)
#     tl = range(l0, l1; length=100)
#     ptau = @. exp(-3.6e-3*(tl/1216)^3.46)
#     out3 = mean(ptau)
#     return SVector{3,T}(out1, out2, out3)
# end


struct Inoue2014IGM <: IGMAttenuation end