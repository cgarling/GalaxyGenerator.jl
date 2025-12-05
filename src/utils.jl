# """
#     distance_modulus(distance)
# Finds distance modulus for distance in parsecs.

# ```math
# μ = 5 \\times \\log_{10}(d) - 5
# ```
# """
# distance_modulus(distance) = 5 * log10(distance) - 5 

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
    interp_lin(x, y, t; extrapolate=false)

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
function interp_lin(x::AbstractVector, y::AbstractVector, t; extrapolate=false)
    @argcheck length(x) == length(y) "x and y must have the same length"
    if t < x[1]
        if extrapolate
            return y[1]  # Return the first value of y
        else
            throw(DomainError("Requested interpolation location outside bounds."))
        end
    elseif t > x[end]
        if extrapolate
            return y[end]  # Return the last value of y
        else
            throw(DomainError("Requested interpolation location outside bounds."))
        end
    end
    # Find the interval that contains t
    i = searchsortedlast(x, t)
    x0, x1 = x[i], x[i+1]
    y0, y1 = y[i], y[i+1]
    return y0 + (y1 - y0) * (t - x0) / (x1 - x0)
end

"""
    interp_log(x, y, t; extrapolate=false)

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
function interp_log(x::AbstractVector, y::AbstractVector, t; extrapolate=false)
    @argcheck length(x) == length(y) "x and y must have the same length"
    if t < x[1]
        if extrapolate
            return y[1]  # Return the first value of y
        else
            throw(DomainError("Requested interpolation location outside bounds."))
        end
    elseif t > x[end]
        if extrapolate
            return y[end]  # Return the last value of y
        else
            throw(DomainError("Requested interpolation location outside bounds."))
        end
    end
    # Find the interval that contains t
    i = searchsortedlast(x, t)
    x0, x1 = log(x[i]), log(x[i+1])
    y0, y1 = y[i], y[i+1]
    t = log(t)
    return y0 + (y1 - y0) * (t - x0) / (x1 - x0)
end

"""
    merge_add(x1, x2, y1, y2)

Merge two SED-like templates `(x1,y1)` and `(x2,y2)` into a single `(x, y)` pair by summing
the values where the wavelength grids overlap. If `x1 == x2`, the function simply returns `x1` and `y1 .+ y2`.

Behavior:
- Inputs must have length(x) == length(y) for each pair and should be sorted ascending.
- When an x from one vector falls strictly before the first x of the other vector,
  the other y is not extrapolated (no contribution).
- When an x lies between two x values of the other vector, linear interpolation is used.
- When x values are equal, the y values are summed directly.

Returns `(xout, yout)`.
"""
function merge_add(x1::AbstractVector{T1}, x2::AbstractVector{T2},
                   y1::AbstractVector{U1}, y2::AbstractVector{U2}) where {T1,T2,U1,U2}

    if x1 == x2
        return x1, y1 .+ y2
    end
    @assert length(x1) == length(y1) "x1 and y1 must have the same length"
    @assert length(x2) == length(y2) "x2 and y2 must have the same length"
    @assert issorted(x1) "x1 must be sorted ascending"
    @assert issorted(x2) "x2 must be sorted ascending"

    Tx = promote_type(T1, T2)
    Ty = promote_type(U1, U2)

    n1 = length(x1)
    n2 = length(x2)

    xout = Vector{Tx}()
    sizehint!(xout, n1 + n2)
    yout = Vector{Ty}()
    sizehint!(yout, n1 + n2)

    i1 = 1
    i2 = 1

    # generic linear interpolation
    interp(y0, y1, x0, x1, x) = x1 == x0 ? y0 : y0 + (y1 - y0) * ((x - x0) / (x1 - x0))

    while i1 <= n1 || i2 <= n2
        if i1 > n1
            push!(xout, convert(Tx, x2[i2]))
            push!(yout, convert(Ty, y2[i2]))
            i2 += 1
        elseif i2 > n2
            push!(xout, convert(Tx, x1[i1]))
            push!(yout, convert(Ty, y1[i1]))
            i1 += 1
        else
            x_a = x1[i1]
            x_b = x2[i2]

            if x_a < x_b
                y_sum = y1[i1]
                if i2 != 1
                    y_sum = y_sum + interp(y2[i2-1], y2[i2], x2[i2-1], x2[i2], x_a)
                end
                push!(xout, convert(Tx, x_a))
                push!(yout, convert(Ty, y_sum))
                i1 += 1

            elseif x_a > x_b
                y_sum = y2[i2]
                if i1 != 1
                    y_sum = y_sum + interp(y1[i1-1], y1[i1], x1[i1-1], x1[i1], x_b)
                end
                push!(xout, convert(Tx, x_b))
                push!(yout, convert(Ty, y_sum))
                i2 += 1

            else
                push!(xout, convert(Tx, x_a))
                push!(yout, convert(Ty, y1[i1] + y2[i2]))
                i1 += 1
                i2 += 1
            end
        end
    end

    return xout, yout
end