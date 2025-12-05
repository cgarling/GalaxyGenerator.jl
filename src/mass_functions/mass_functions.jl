"""Module containing modules for cosmological stellar mass functions."""
module MassFunctions

using ..GalaxyGenerator: inverse_cdf!

using ArgCheck: @argcheck, @check
using Compat: logrange
using Random: Random, default_rng, AbstractRNG
using IrrationalConstants: logten

"""Abstract type for cosmological stellar mass functions (e.g., Schechter)."""
abstract type AbstractMassFunction{T} end
function Random.rand(rng::Random.AbstractRNG, s::AbstractMassFunction)
    icdf = s.icdf
    if isnothing(icdf)
        error("Provided SchechterMassFunction has no inverse CDF buffer; please create new instance with `mmin` and `mmax` arguments.")
    else
        icdf(rand(rng))
    end
end
function Random.rand(rng::Random.AbstractRNG, s::AbstractMassFunction, dims::Dims)
    return reshape([rand(rng, s) for _ in 1:prod(dims)], dims)
end

include("Schechter.jl")
export SchechterMassFunction, DoubleSchechterMassFunction

end # module