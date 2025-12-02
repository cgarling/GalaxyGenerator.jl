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

    for i in 1:n1
        @views if lam[:, i, :] != lam[:, :, i]
            return false
        end
    end
    # for i in 1:n1
    #     for j in 1:n2
    #         @views if lam[:,i,j] != lam[:,1,1]
    #             return false
    #         end
    #     end
    # end

    return true
end

struct OptLib
    # lam[uv, vj, λ]
    lam::Array{Float32,3}
    # sed[uv, vj, λ] (per unit stellar mass; scale by 10^m)
    sed::Array{Float32,3}
    # use[uv, vj]
    use::BitMatrix
    # av[uv, vj]
    av::Matrix{Float32}
    # bin descriptors for U-V and V-J. These may be either:
    #  - a vector of bin edges of length nbins+1, OR
    #  - a vector of bin centers of length nbins
    buv::Vector{Float32}   # U-V bin definition
    bvj::Vector{Float32}   # V-J bin definition
    # derived index map built by build_idxmap!
    idxmap::Matrix{Int}   # idxmap[u,v] -> used-template index (1-based) or 0 if unused
    used_coords::Vector{Tuple{Int,Int}} # reverse map: used_coords[ised] = (u,v)
end
function OptLib(fname::AbstractString)
    FITS(fname, "r") do f
        hdu = f[2]
        lam = read(hdu, "LAM")[:,:,:,1]
        # @argcheck map(==(lam
        sed = read(hdu, "SED")[:,:,:,1]
        buv = read(hdu, "BUV")# [:,:,1]
        buv = vcat(buv[:,1,1], buv[end,2,1]) # Reduce to vector of bin edges, length nbins + 1
        @argcheck issorted(buv)
        bvj = read(hdu, "BVJ")[:,:,1]
        bvj = vcat(bvj[:,1,1], bvj[end,2,1]) # Reduce to vector of bin edges, length nbins + 1
        @argcheck issorted(bvj)
        use = convert.(Bool, read(hdu, "USE")[:,:,1])
        av = read(hdu, "AV")[:,:,1]

        idxmap, used = build_idxmap(use)
        return OptLib(lam, sed, use, av, buv, bvj, idxmap, used)
    end
end

# Build idxmap and used_coords. Must be called once after creating OptLib (or when optlib.use changes).
# function build_idxmap!(opt::OptLib)
#     nu, nv = size(opt.use)
#     opt.idxmap = zeros(Int, nu, nv)
#     used = Vector{Tuple{Int,Int}}()
#     k = 1
#     for u in 1:nu, v in 1:nv
#         if opt.use[u,v]
#             opt.idxmap[u,v] = k
#             push!(used, (u,v))
#             k += 1
#         else
#             opt.idxmap[u,v] = 0
#         end
#     end
#     opt.used_coords = used
#     return opt
# end
function build_idxmap(USE::BitMatrix)
    nu, nv = size(USE)
    idxmap = zeros(Int, nu, nv)
    used = Vector{Tuple{Int,Int}}()
    k = 1
    for u in 1:nu, v in 1:nv
        if USE[u,v]
            idxmap[u,v] = k
            push!(used, (u,v))
            k += 1
        else
            idxmap[u,v] = 0
        end
    end
    return idxmap, used
end

# Convert a vector of bin edges / centers into a function that finds bin index.
# Returns 0 if value is outside all bins.
# Behavior:
# - If `edges` length == nbins+1 => interpret as explicit edges: bins are [edges[i], edges[i+1])
#   for i=1..nbins-1 and [edges[end-1], edges[end]] for last bin (right inclusive).
# - If `edges` length == nbins => interpret as bin centers. Bins are midpoints between centers.
# function find_bin(value::Real, edges::AbstractVector{<:Real})
#     n = length(edges)
#     if n == 0
#         return 0
#     end
#     # If edges are edges (nbins+1)
#     # detect if the length is at least 2 and represents edges by checking monotonicity
#     # (we accept either representation - edges or centers)
#     # Here we decide by whether length(edges) == nbins+1 is plausible (we can't know nbins).
#     # We'll choose: if length(edges) >= 2 and the values could be edges,
#     # we check both possibilities: try edges-as-edges first if n >= 2.
#     # For robustness, support both via heuristics below.

#     # Heuristic: if there are more than 2 entries and spacing is non-uniform, treat as edges.
#     if n >= 2
#         # if edges likely represent bin-edges (length bins+1)
#         # we'll treat as edges when consecutive spacing suggests boundaries:
#         # (this is safe in practice for typical UVJ grids).
#         # Implement direct "edges" behavior:
#         for i in 1:(n-1)
#             left = edges[i]
#             right = edges[i+1]
#             if (i < n-1 && value >= left && value < right) || (i == n-1 && value >= left && value <= right)
#                 return i
#             end
#         end
#     end

#     # If we didn't find it as edges, treat as centers:
#     # Build midpoints
#     if n == 1
#         # single center => only one bin
#         return 1
#     end
#     mid = Vector{Float64}(undef, n-1)
#     for i in 1:(n-1)
#         mid[i] = 0.5*(edges[i] + edges[i+1])
#     end
#     # bins:
#     # (-Inf, mid[1]) -> bin1
#     if value < mid[1]
#         return 1
#     end
#     for i in 2:(n-1)
#         if value < mid[i]
#             return i
#         end
#     end
#     return n   # value >= mid[end] -> last bin
# end
function find_bin(value::Real, edges::AbstractVector{<:Real})
    Base.require_one_based_indexing(edges)
    n = length(edges)
    n < 2 && return 1 # Degenerate case
    # Binary search: finds first index j where edges[j] ≥ value
    j = searchsortedfirst(edges, value)
    if (edges[j] ≈ value) || (j == 1) # value exact match, or value < first edge
        return j
    elseif j > n
        return n - 1 # value ≥ last edge
    else
        return j - 1 # normal case
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

# """
#     get_sed_uvj(uv::AbstractVector{<:Real}, vj::AbstractVector{<:Real}, optlib::OptLib; doprint=false)

# Map arrays of rest-frame UV (uv) and VJ (vj) colors to template indices (ised) and AV offsets (avs).

# Returns (ised::Vector{Int}, avs::Vector{Float32}).

# ised is a 1-based index into the compact list of usable templates (matching optlib.av).
# If no valid template could be found for an object, ised[i] == 0 and avs[i] == NaN.
# """
# function get_sed_uvj(uv::AbstractVector{<:Real}, vj::AbstractVector{<:Real}, optlib::OptLib; doprint::Bool=false)
#     nobj = length(uv)
#     @assert length(vj) == nobj "uv and vj must have same length"

#     # Ensure idxmap built
#     if isempty(optlib.idxmap)
#         error("optlib.idxmap is empty; call build_idxmap!(optlib) before using get_sed_uvj")
#     end

#     nu, nv = size(optlib.use)
#     sed = zeros(Int, nobj)
#     avs = fill(Float32(NaN), nobj)

#     for i in 1:nobj
#         u = find_bin(uv[i], optlib.buv)
#         v = find_bin(vj[i], optlib.bvj)

#         if u == 0 || v == 0 || u > nu || v > nv
#             # out of range
#             sed[i] = 0
#             avs[i] = Float32(NaN)
#             if doprint
#                 @info "object $i: uv=$(uv[i]) vj=$(vj[i]) -> out of UVJ range"
#             end
#             continue
#         end

#         if optlib.use[u, v]
#             ised = optlib.idxmap[u, v]
#             sed[i] = ised
#             avs[i] = optlib.av[ised]
#             if doprint
#                 @info "object $i -> exact cell ($u,$v) -> ised=$ised"
#             end
#         else
#             uu, vv, ok = astar_find(optlib.use, u, v)
#             if !ok
#                 # no usable template anywhere
#                 sed[i] = 0
#                 avs[i] = Float32(NaN)
#                 if doprint
#                     @info "object $i: no usable template found for cell ($u,$v)"
#                 end
#             else
#                 ised = optlib.idxmap[uu, vv]
#                 sed[i] = ised
#                 avs[i] = optlib.av[ised]
#                 if doprint
#                     @info "object $i: cell ($u,$v) empty -> nearest ($uu,$vv) ised=$ised"
#                 end
#             end
#         end
#     end

#     return sed, avs
# end
"""
    get_sed_uvj(uv, vj, optlib::OptLib; doprint=false)

Map rest-frame UV (`uv`) and VJ (`vj`) colors to template indices (`ised`). `ised` is a 1-based index into the compact list of usable templates (matching `optlib.av`). If no valid template could be found for an object, ised[i] == 0 and avs[i] == NaN.
"""
function get_sed_uvj(uv, vj, optlib::OptLib; doprint::Bool=false)
    u = find_bin(uv, optlib.buv)
    v = find_bin(vj, optlib.bvj)

    if optlib.use[u, v]
        ised = optlib.idxmap[u, v]
    else
        uu, vv, ok = astar_find(optlib.use, u, v)
        if !ok
            error("No valid template found for U-V = $uv, V-J=$vj.")
        else
            ised = optlib.idxmap[uu, vv]
        end
    end
    return ised
end

"""
    get_opt_sed(optlib::OptLib, Mstar::Real, ised::Int)

Return `(lam, sed)` for the requested usable-template index `ised` (1-based).
- `lam` and `sed` are Vector{Float32}.
- `sed` is scaled by `Mstar`, provided in solar masses (i.e. templates are per unit stellar mass).
"""
function get_opt_sed(uv, vj, optlib::OptLib, Mstar::Real)
    ised = get_sed_uvj(uv, vj, optlib)
    @argcheck ised > 0
    if ised > length(optlib.used_coords)
        error("`ised` $ised out of range")
    end
    (u, v) = optlib.used_coords[ised]
    lam = optlib.lam[:, u, v]
    sed = optlib.sed[:, u, v]
    sed .*= Mstar
    Av = optlib.av[u, v]
    return lam, sed, Av
end
