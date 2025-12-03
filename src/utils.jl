"""
    inverse_cdf!(y, x)
Given `x` and `y` arrays that fulfill `f(x) = y`, compute and return an interpolator for the inverse CDF of the function `f(x)`. `y` is mutated in-place to contain the CDF.
"""
function inverse_cdf!(y, x)
    @argcheck length(x) == length(y)
    Base.require_one_based_indexing(y, x)
    cdf = y # in place
    cumsum!(@view(cdf[2:end]), @views (cdf[2:end] .+ cdf[1:end-1]) ./ 2 .* diff(x))
    cdf[1] = 0
    cdf ./= last(cdf)
    return AkimaInterpolation(x, cdf)
end
# function inverse_cdf(pdf, bins)
#     @argcheck
#     pdf = s.(bins) # Evaluate at bins
#     cdf = pdf # in place
#     cumsum!(@view(cdf[2:end]), @views (cdf[2:end] .+ cdf[begin:end-1]) ./ 2 .* diff(bins))
#     cdf[1] = 0
#     cdf ./= last(cdf)
#     return AkimaInterpolation(bins, cdf)
# end

"""
    interp_lin(x, y, t)

Linearly interpolates the value of `y` as a function of `x` at the point `t`.

# Arguments
- `x`: Sorted array of independent variables. *Must be pre-sorted; this function does not check*.
- `y`: Array of dependent variables, the same size as `x`.
- `t`: The point at which to interpolate.

# Returns
The interpolated value of `y` corresponding to `t`.

# Notes
- If `t` is outside the range of `x`, the function throws an error.
"""
function interp_lin(x::AbstractVector, y::AbstractVector, t::Real)
    @argcheck length(x) == length(y) "x and y must have the same length"
    if (t <= x[1]) || (t >= x[end])
        throw(DomainError("Requested interpolation location outside bounds."))
    end
    # Find the interval that contains t
    i = searchsortedlast(x, t)
    x0, x1 = x[i], x[i+1]
    y0, y1 = y[i], y[i+1]
    return y0 + (y1 - y0) * (t - x0) / (x1 - x0)
end

"""
    interp_log(x, y, t)

Linearly interpolates the value of `y` as a function of `log(x)` at the point `log(t)`. For some types of data, you would rather interpolate in `log(x)` than `x` because `log(x)` is smoother, reducing interpolation error. This function computes only the logarithms of the `x` values that it needs for the interpolation, avoiding the need to precompute `log.(x)`.

# Arguments
- `x`: Sorted array of independent variables. *Must be pre-sorted; this function does not check*.
- `y`: Array of dependent variables, the same size as `x`.
- `t`: The point at which to interpolate.

# Returns
The interpolated value of `y` corresponding to `t`.

# Notes
- If `t` is outside the range of `x`, the function throws an error.
"""
function interp_log(x::AbstractVector, y::AbstractVector, t::Real)
    @argcheck length(x) == length(y) "x and y must have the same length"
    if (t <= x[1]) || (t >= x[end])
        throw(DomainError("Requested interpolation location outside bounds."))
    end
    # Find the interval that contains t
    i = searchsortedlast(x, t)
    x0, x1 = log(x[i]), log(x[i+1])
    y0, y1 = y[i], y[i+1]
    t = log(t)
    return y0 + (y1 - y0) * (t - x0) / (x1 - x0)
end