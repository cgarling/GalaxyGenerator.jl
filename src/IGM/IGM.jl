module IGM

using ArgCheck: @argcheck, @check
using DelimitedFiles: readdlm
import Unitful as u

export IGMAttenuation, NoIGM, Madau1995IGM, Inoue2014IGM, transmission, tau

abstract type IGMAttenuation end
Base.Broadcast.broadcastable(m::IGMAttenuation) = Ref(m)

# Generic methods
"""
    transmission(model::IGMAttenuation, z, λ_r)
Returns the IGM transmission for light emitted at rest wavelength `λ_r` (in Angstroms) at redshift `z` given the provided IGM transmission model `model`. Defined as `exp(-τ)` where `τ` is the optical depth. Generic implementation is `transmission(tau(model, z, λ_r))`.
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
Model for no IGM absorption. `transmission(::NoIGM, z, λ_r)` always returns `one(z)`, adding no attenuation.
"""
struct NoIGM <: IGMAttenuation end

"""
    tau(model::IGMAttenuation, z, λ_r)
Returns the IGM optical depth for light emitted at rest wavelength `λ_r` (in Angstroms) at redshift `z` given the provided IGM transmission model `model`.
"""
function tau end
tau(::NoIGM, z, λ_r) = zero(z)
transmission(::NoIGM, z, λ_r) = one(z) # Define this directly to avoid the generic exp(-tau) method

#################
# Madau1995 model 
"""
    Madau1995IGM()

Implements the analytic IGM transmission model from [Madau1995](@citet).
Computing the [`transmission`](@ref GalaxyGenerator.IGM.transmission) takes roughly 444 ns.
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
    lylim   = 911.75
    a_metal = 0.0017

    z1   = 1 + z
    λ_o  = λ_r * z1
    xc   = λ_o / lylim
    # if z1 > xc / 0.8
    #     @warn "Madau1995IGM model has poor performance at short wavelengths. As a precaution, this function will return the transmission value evaluated at λ_rest = 729.4 Å." maxlog=1
    #     return transmission(Madau1995IGM(), z, lylim * 0.8) # z1 = xc = λ_r * z1 / lylim, solve for λ_r
    # end
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

#########################################
# Inoue2014 model #######################

"""
    Inoue2014IGM()

Implements the analytic IGM transmission model from [Inoue2014](@citet). This model accounts for 
Lyman-α forest, dampled Lyman-α, Lyman-series, and Lyman-continuum absorption by the intergalactic medium (IGM).
Computing the [`transmission`](@ref GalaxyGenerator.IGM.transmission) takes roughly 675 ns.

Because this model takes some time to initialize and takes no arguments, we provide a pre-initialized instance as the constant `GalaxyGenerator.IGM.Inoue2014`.
"""
struct Inoue2014IGM <: IGMAttenuation
    lam::Vector{Float32}   # wavelengths (Angstroms) for Lyman series lines
    ALAF::Matrix{Float32}   # LAF coefficients (N x 3)
    ADLA::Matrix{Float32}   # DLA coefficients (N x 2)
end

function Inoue2014IGM()
    laf_file = joinpath(@__DIR__, "data", "Inoue14", "LAFcoeff.txt")
    dla_file = joinpath(@__DIR__, "data", "Inoue14", "DLAcoeff.txt")

    laf = readdlm(laf_file, Float32)
    dla = readdlm(dla_file, Float32)
    lam = laf[:, 2]
    @check lam == dla[:, 2] "Wavelengths in LAF and DLA files do not match."
    return Inoue2014IGM(lam, laf[:, 3:end], dla[:, 3:end])
end

"""Pre-initialized [`Inoue2014IGM`](@ref) model instance."""
const Inoue2014 = Inoue2014IGM()

# Lyman series optical depth by the DLA component
function tLSDLA(zS::Real, lobs::Real, line_λ::AbstractVector{<:Real}, A::AbstractMatrix{<:Real})
    z1DLA = 2.0
    τ = 0.0
    for j in eachindex(line_λ)
        lamj = line_λ[j]
        if (lobs < lamj * (1 + zS)) && (lobs > lamj)
            if lobs < lamj * (1 + z1DLA)
                τ += A[j, 1] * (lobs / lamj)^2
            else
                τ += A[j, 2] * (lobs / lamj)^3
            end
        end
    end
    return τ
end

# Lyman series optical depth by the LAF component
function tLSLAF(zS::Real, lobs::Real, line_λ::AbstractVector{<:Real}, A::AbstractMatrix{<:Real})
    z1LAF = 1.2
    z2LAF = 4.7
    τ = 0.0
    for j in eachindex(line_λ)
        lamj = line_λ[j]
        if (lobs < lamj * (1 + zS)) && (lobs > lamj)
            if lobs < lamj * (1 + z1LAF)
                τ += A[j, 1] * (lobs / lamj)^1.2
            elseif lobs < lamj * (1 + z2LAF)
                τ += A[j, 2] * (lobs / lamj)^3.7
            else
                τ += A[j, 3] * (lobs / lamj)^5.5
            end
        end
    end
    return τ
end

# Lyman continuum optical depth by the DLA component
function tLCDLA(zS::Real, lobs::Real)
    z1DLA = 2.0
    lamL = 911.8
    if lobs > lamL * (1 + zS)
        return 0.0
    elseif zS < z1DLA
        return (0.2113 * (1 + zS)^2 - 0.07661 * (1 + zS)^2.3 * (lobs / lamL)^(-0.3) - 0.1347 * (lobs / lamL)^2)
    elseif lobs > lamL * (1 + z1DLA)
        return (0.04696 * (1 + zS)^3 - 0.01779 * (1 + zS)^3.3 * (lobs / lamL)^(-0.3) - 0.02916 * (lobs / lamL)^3)
    else
        return (0.6340 + 0.04696 * (1 + zS)^3 - 0.01779 * (1 + zS)^3.3 * (lobs / lamL)^(-0.3) - 0.1347 * (lobs / lamL)^2 - 0.2905 * (lobs / lamL)^(-0.3))
    end
end

# Lyman continuum optical depth by the LAF component
function tLCLAF(zS::Real, lobs::Real)
    z1LAF = 1.2
    z2LAF = 4.7
    lamL = 911.8
    if lobs > lamL * (1 + zS)
        return 0.0
    elseif zS < z1LAF
        return 0.3248 * ((lobs / lamL)^1.2 - (1 + zS)^(-0.9) * (lobs / lamL)^2.1)
    elseif zS < z2LAF
        if lobs > lamL * (1 + z1LAF)
            return 0.02545 * ((1 + zS)^1.6 * (lobs / lamL)^2.1 - (lobs / lamL)^3.7)
        else
            return 0.02545 * (1 + zS)^1.6 * (lobs / lamL)^2.1 + 0.3248 * (lobs / lamL)^1.2 - 0.2496 * (lobs / lamL)^2.1
        end
    else
        if lobs > lamL * (1 + z2LAF)
            return 0.0005221 * ((1 + zS)^3.4 * (lobs / lamL)^2.1 - (lobs / lamL)^5.5)
        elseif lobs > lamL * (1 + z1LAF)
            return 0.0005221 * (1 + zS)^3.4 * (lobs / lamL)^2.1 + 0.2182 * (lobs / lamL)^2.1 - 0.02545 * (lobs / lamL)^3.7
        else
            return 0.0005221 * (1 + zS)^3.4 * (lobs / lamL)^2.1 + 0.3248 * (lobs / lamL)^1.2 - 0.0314 * (lobs / lamL)^2.1
        end
    end
end

# Optical depth function for Inoue2014 model.
# Input: λ_r in angstroms (rest-frame).
function tau(model::Inoue2014IGM, z::Real, λ_r::Real)
    # lobs = λ_r * (1 + z)
    # τ = tLSLAF(z, lobs, model.lam, model.ALAF) +
    #     tLSDLA(z, lobs, model.lam, model.ADLA) +
    #     tLCLAF(z, lobs) +
    #     tLCDLA(z, lobs)
    # return τ
    τ = if λ_r > 1215.67
        0.0
    else
        lobs = λ_r * (1 + z)
        tLSLAF(z, lobs, model.lam, model.ALAF) +
        tLSDLA(z, lobs, model.lam, model.ADLA) +
        tLCLAF(z, lobs) +
        tLCDLA(z, lobs)
    end
    return τ
end

#########################
# Generic unitful methods
for T in (NoIGM, Madau1995IGM, Inoue2014IGM)
    @eval transmission(s::$T, z, λ_r::u.Length)= transmission(s, z, u.ustrip(u.angstrom, λ_r))
    @eval tau(s::$T, z, λ_r::u.Length)= tau(s, z, u.ustrip(u.angstrom, λ_r))
end

end # module IGM
