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