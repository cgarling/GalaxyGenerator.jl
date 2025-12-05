"""Module for handling emission lines in galaxy SEDs."""
module EmissionLines

using IrrationalConstants: sqrt2
using SpecialFunctions: erf
import Unitful as u

export add_emission_line!

"""
    add_emission_line!(lam, sed, λ0, L_line, vdisp_kms)

Add a single emission line to a continuum `sed` in-place.

Arguments
- `lam::AbstractVector{<:Real}` : monotonically increasing wavelength grid (same units as `λ0`, e.g. µm)
- `sed::AbstractVector{<:Real}` : input spectrum on the `lam` grid (will be modified in-place)
- `λ0::Real`                    : rest wavelength of the line (same units as `lam`, e.g. µm)
- `L_line::Real`                : line luminosity in whatever unit your SED uses
- `vdisp_kms::Real`             : Gaussian sigma in km/s (velocity dispersion). If zero or negative,
                                  the function does nothing.

Implementation notes
- The line is modeled as a Gaussian profile with sigma = λ0 * (vdisp / c)
  (c in km/s). For each wavelength bin the function computes the fraction of
  the Gaussian area falling in the bin using the error function, and then
  converts that area into an average per-wavelength value before adding it
  to the `sed` array.


# Examples
```jldoctest
julia> using GalaxyGenerator.EmissionLines

julia> lam = 0.64:1e-4:0.66; # µm grid

julia> sed = zeros(length(lam)); # empty continuum

julia> add_emission_line!(lam, sed, 0.65628, 1.0, 100.0)  # L_line=1.0, vdisp=100 km/s

julia> area = sum(0.5*(sed[1:end-1] .+ sed[2:end]) .* diff(lam));  # trapezoidal integral

julia> isapprox(area, 0.65628; atol=1e-3)
true
```
"""
function add_emission_line!(lam::AbstractVector{A}, sed::AbstractVector,
                            λ0, L_line, vdisp_kms) where A

    @assert length(lam) == length(sed) "lam and sed must have the same length"
    n = length(lam)
    if n == 0
        return
    end

    # speed of light in km/s
    c_km_s = A(2.99792458e5)

    σ = λ0 * (vdisp_kms / c_km_s)
    if !(isfinite(σ) && σ > 0)
        return
    end

    # consider only +/- 10 sigma around the line center
    low  = λ0 - 10 * σ
    high = λ0 + 10 * σ

    # find indices that overlap that interval (1-based)
    b0 = searchsortedfirst(lam, low)
    b1 = searchsortedlast(lam, high)
    b0 = max(1, b0)
    b1 = min(n, b1)
    if b0 > b1
        return
    end

    # precompute bin edges (left/right) for the affected bins
    m = b1 - b0 + 1
    llow = Vector{A}(undef, m)
    lup  = Vector{A}(undef, m)

    for k in 1:m
        i = b0 + k - 1
        # left edge
        if i == 1
            # extrapolate first left edge using first bin width
            llow[k] = lam[i] - (lam[i+1] - lam[i]) / 2
        else
            llow[k] = (lam[i-1] + lam[i]) / 2
        end

        # right edge
        if i == n
            lup[k] = lam[i] + (lam[i] - lam[i-1]) / 2
        else
            lup[k] = (lam[i] + lam[i+1]) / 2
        end
    end

    # total "area" associated to the line
    area = L_line * λ0
    sqrt2σ = sqrt2 * σ

    # add the average line contribution per bin to sed
    for k in 1:m
        l = llow[k]
        u = lup[k]
        # fraction of Gaussian area in [l,u]
        frac = (erf((u - λ0) / sqrt2σ) - erf((l - λ0) / sqrt2σ)) / 2
        # area of bin
        a_bin = area * frac
        # convert to average per-wavelength value over the bin
        val = a_bin / (u - l)
        sed[b0 + k - 1] += val
    end

    return nothing
end

# function add_emission_line!(lam::AbstractVector{<:u.Quantity}, sed::AbstractVector{T},
#                             λ0::u.Length, L_line::u.Quantity, vdisp_kms::u.Velocity) where {T<:u.Quantity}
#     return add_emission_line!(u.ustrip(u.angstrom, lam),
#                                u.ustrip(u.erg / u.s / u.cm^2 / u.angstrom, sed),
#                                u.ustrip(u.angstrom, λ0),
#                                u.ustrip(u.erg / u.s / u.cm^2 / u.angstrom, L_line),
#                                u.ustrip(u.km / u.s, vdisp_kms))
# end

end # module EmissionLines