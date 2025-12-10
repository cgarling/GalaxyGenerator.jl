# Utility for selecting optical SED templates from rest-frame UVJ colors
#
# This file provides:
# - an OptLib container shape expected by the rest of the generator
# - helpers to build an index of "usable" templates
# - find_bin / in-bin logic that is tolerant to either edges or bin-centers input
# - astar search to find the nearest usable template when the exact UVJ cell is empty
# - get_sed_uvj(uv, vj, optlib; doprint=false) -> (isedices, avs)
# - get_opt_sed(optlib, m, ised) -> (lam, sed)  (scales template by 10^m)
#
# Notes about conventions:
# - optlib.use is expected to be a Bool matrix (nu x nv) marking which UVJ cells have a template.
# - optlib.sed and optlib.lam are expected to be 3D arrays (nu x nv x nλ) holding wavelength
#   grids and SED values per cell. The wavelength grid CAN vary per (u,v) cell (we keep it per-cell).
# - optlib.av is a 1D Vector with one value per usable template (the same indexing used by `idxmap`).
# - The returned ised indices are 1-based indices into the compact list of usable templates
#   (matching optlib.av), i.e. the same "flat" indexing used in the original C++ code.
# 
# Units:
# Looks like wavelengths (lam) are in microns
# SEDs are in solar luminosity per unit stellar mass 
# (L_sun per unit-wavelength per Msun — effectively L_sun/Δλ per Msun); need to convert to physical,
# EGG converts to μJy 

"""
    lsun2Jy(lam, lum)
Convert solar luminosities to Jansky at distance 10 pc. Used for normalizing SEDs from EGG.
"""
function lsun2Jy(lam, lum)
    d = 1e-5                   # 10 pc in Mpc, to place flux in absolute units
    Mpc = 3.085677581491367e22 # [m/Mpc]
    Lsol = 3.828e26            # [W/Lsol], IAU Resolution B3
    Jy = 1e26                  # [Jy/(W.m-2.Hz-1)]
    c = 2.99792458e18          # [angstrom.s-1]
    factor = Jy * Lsol / (c * 4 * π * Mpc * Mpc)
    return factor * lam * lum / d^2
end

"""
    Jy2cgs(lam, f_nu)
Convert flux density from Jansky to CGS flux units (erg/s/cm²/Å) at wavelength `lam` (in angstrom).
"""
function Jy2cgs(lam, f_nu)
    c = 2.99792458e8 # [m/s]
    return c / 10^8 / 10^5 * f_nu / lam^2
end

"""
    lsun2cgs(lam, lum)
Convert solar luminosities to CGS flux units (erg/s/cm²/Å) at 10 pc. Used for normalizing SEDs from EGG.
"""
lsun2cgs(lam, lum) = Jy2cgs(lam, lsun2Jy(lam, lum))

"""
    _check_lam(lam)
Returns true if `lam[:,i,:] == lam[:,:,i]` for all valid indices `i`, indicating that the array is mirrored. Check that all the optical SED library files have uniform lambda vectors.
    
Apparently `opt_lib_fast_hd.fits` and `opt_lib_fast_hd_noigm.fits` do not...

```julia
using GalaxyGenerator
using FITSIO
files = readdir(joinpath(splitdir(pathof(GalaxyGenerator))[1], "EGG", "data"); join=true)
optlibs = filter(Base.Fix1(occursin, "opt_lib"), files)
for file in optlibs
    FITS(file, "r") do f
        lam = read(f[2], "LAM")[:,:,:,1]
        println(GalaxyGenerator.EGG._check_lam(lam))
    end
end
```
"""
function _check_lam(lam)
    _, n1, n2 = size(lam)
    n1 == n2 || return false  # must be square in the last two dims

    # for i in 1:n1
    #     @views if lam[:, i, :] != lam[:, :, i]
    #         return false
    #     end
    # end
    for i in 1:n1
        for j in 1:n2
            @views if lam[:,i,j] != lam[:,1,1]
                return false
            end
        end
    end

    return true
end

# Returns the correction to the mass-to-light ratio in dex
get_m2l_cor(z) = interp_lin(SVector(0.0,0.45,1.3,6.0,8.0), SVector(0.15,0.15,0.0,0.0,-0.6), z)

struct OptLib
    # lam[uv, vj, λ]
    lam::Array{Float32,3}
    # sed[uv, vj, λ] (per unit stellar mass; scale by 10^m)
    sed::Array{Float32,3}
    # use[uv, vj]
    use::BitMatrix
    # av[uv, vj]
    av::Matrix{Float32}
    buv::Vector{Float32}   # U-V bin definition, length(nbins) + 1
    bvj::Vector{Float32}   # V-J bin definition, length(nbins) + 1
end
function OptLib(fname::AbstractString=joinpath(@__DIR__, "data", "opt_lib_fast_noigm.fits"))
    FITS(fname, "r") do f
        hdu = f[2]
        lam = read(hdu, "LAM")[:,:,:,1]
        lam .*= 1f4 # convert microns to angstroms
        sed = read(hdu, "SED")[:,:,:,1]
        @. sed = lsun2cgs(lam, sed) # convert from Lsun/Msun to erg/s/cm²/Å per Msun
        buv = read(hdu, "BUV")
        buv = vcat(buv[:,1,1], buv[end,2,1]) # Reduce to vector of bin edges, length nbins + 1
        @argcheck issorted(buv)
        bvj = read(hdu, "BVJ")[:,:,1]
        bvj = vcat(bvj[:,1,1], bvj[end,2,1]) # Reduce to vector of bin edges, length nbins + 1
        @argcheck issorted(bvj)
        use = convert.(Bool, read(hdu, "USE")[:,:,1])
        av = read(hdu, "AV")[:,:,1]

        return OptLib(lam, sed, use, av, buv, bvj)
    end
end

# astar_find: find nearest (u,v) with use[u,v]==true, starting from (u0,v0)
# returns (u_found, v_found, true) or (0,0,false) if none found.
function astar_find(use::AbstractMatrix{<:Bool}, u0::Int, v0::Int)
    nu, nv = size(use)
    if use[u0, v0]
        return (u0, v0, true)
    end
    maxr = max(nu, nv)
    for r in 1:maxr
        for du in -r:r
            dv = r - abs(du)
            for sgn in (-1, 1)
                v = v0 + sgn*dv
                u = u0 + du
                if 1 <= u <= nu && 1 <= v <= nv
                    if use[u,v]
                        return (u,v,true)
                    end
                end
            end
        end
    end
    return (0,0,false)
end

"""
    get_sed_uvj(uv, vj, optlib::OptLib)

Map rest-frame U-V (`uv`) and V-J (`vj`) colors to template indices (`ised`). `ised` is a 1-based index into the compact list of usable templates (matching `optlib.av`). If no valid template could be found for an object, ised[i] == 0 and avs[i] == NaN.
"""
function get_sed_uvj(uv, vj, optlib::OptLib)
    u = find_bin(uv, optlib.buv)
    v = find_bin(vj, optlib.bvj)

    if optlib.use[u, v]
        return u, v
    else
        u, v, ok = astar_find(optlib.use, u, v)
        if !ok
            error("No valid template found for U-V = $uv, V-J=$vj.")
        else
            return u, v
        end
    end
end

"""
    get_opt_sed(uv, vj, optlib::OptLib)
Retrieves optical SED from `optlib` for galaxy with provided `uv` U-V color and `vj` V-J color. 

Returns
 - `lambda` wavelength in Å
 - `sed` spectral energy distribution in CGS units (erg/s/cm²/Å) at 10 pc (absolute units) normalized per unit stellar mass (Msun)
 - `Av` internal V-band extinction in magnitudes of the selected template. This is output from the SED fitting code used to build the library and this extinction is already applied to the SED.
"""
function get_opt_sed(uv, vj, optlib::OptLib)
    u, v = get_sed_uvj(uv, vj, optlib)
    lam = optlib.lam[:, u, v]   # pre-converted to angstroms in OptLib constructor
    sed = optlib.sed[:, u, v]   # pre-converted to erg/s/cm²/Å per Msun in OptLib constructor
    Av = optlib.av[u, v]
    return lam, sed, Av
end

# Computing magnitude takes ~20μs per filter

# Both libraries ir_lib_ce01.fits and ir_lib_cs17.fits have one λ vector shared for all SEDs
# This struct will be for cs17 library
abstract type IRLib end
struct CS17_IRLib <: IRLib
    lam::Vector{Float32}  # microns
    dust::Matrix{Float32} # DUST[λ, Tdust] (per unit Mdust)
    pah::Matrix{Float32}  # pah[λ, Tdust] (per unit Mdust)
    tdust::Vector{Float32}
    lir_dust::Vector{Float32}
    lir_pah::Vector{Float32}
    l8_dust::Vector{Float32}
    l8_pah::Vector{Float32}
end
function CS17_IRLib(fname::AbstractString=joinpath(@__DIR__, "data", "ir_lib_cs17.fits"))
    FITS(fname, "r") do f
        hdu = f[2]
        lam = read(hdu, "LAM")[:,:,1]
        lam .*= 1f4 # convert microns in file to angstroms
        # Convert dust and pah SEDs from L⊙ / Mdust to cgs erg/s/cm²/Å per Mdust
        dust = read(hdu, "DUST")[:,:,1]
        pah = read(hdu, "PAH")[:,:,1]
        @. dust = lsun2cgs(lam, dust)
        @. pah = lsun2cgs(lam, pah)
        # Check that all wavelength columns are the same, then reduce lam to vector
        @check all(map(==(view(lam, :, 1)), eachcol(lam)))
        lam = lam[:,1]
        @check issorted(lam)
        tdust = read(hdu, "TDUST")[:,1]
        @argcheck issorted(tdust)
        lir_dust = read(hdu, "LIR_DUST")[:,1]
        lir_pah = read(hdu, "LIR_PAH")[:,1]
        l8_dust = read(hdu, "L8_DUST")[:,1]
        l8_pah = read(hdu, "L8_PAH")[:,1]
        return CS17_IRLib(lam, dust, pah, tdust, lir_dust, lir_pah, l8_dust, l8_pah)
    end
end

"""
    get_ir_sed(tdust, irlib::CS17_IRLib)
Retrieves the appropriate IR SED from the Schreiber+2017 library given the dust temperature `tdust` in Kelvin of a galaxy.

Returns
A `NamedTuple` with fields:
 - `lam` wavelength in Å
 - `dust` spectral energy distribution of dust in CGS units (erg/s/cm²/Å) at 10 pc (absolute units) normalized per unit dust mass (Msun)
 - `pah` spectral energy distribution of PAHs in CGS units (erg/s/cm²/Å) at 10 pc (absolute units) normalized per unit dust mass (Msun)
 - `lir_dust` total IR luminosity from dust component in L⊙ per unit dust mass (Msun)
 - `lir_pah` total IR luminosity from PAH component in L⊙ per unit dust mass (Msun)
 - `l8_dust` 8μm luminosity from dust component in L⊙ per unit dust mass (Msun)
 - `l8_pah` 8μm luminosity from PAH component in L⊙ per unit dust mass (Msun)
"""
function get_ir_sed(tdust, irlib::CS17_IRLib)
    i = find_bin(tdust, irlib.tdust)
    # lam is pre-converted to angstrom, dust and pah SEDs to erg/s/cm²/Å per Mdust in CS17_IRLib constructor
    return (lam = copy(irlib.lam), dust = irlib.dust[:,i], pah = irlib.pah[:,i], lir_dust = irlib.lir_dust[i], lir_pah = irlib.lir_pah[i], l8_dust = irlib.l8_dust[i], l8_pah = irlib.l8_pah[i])
end
