# Code for Schechter luminosity/mass functions

"""
    SchechterMassFunction(ϕ, α, Mstar0)
The Schechter mass function model, defined as 

```math
S = \\text{log}(10) \\ \\phi_* \\left( \\frac{M_*}{M_{*,0}} \\right)^{\\alpha+1} \\text{exp} \\left(- \\frac{M_*}{M_{*,0}} \\right)
```

Calling an instance of `SchechterMassFunction` with a stellar mass (in solar masses) will return the value of the mass function. 

    SchechterMassFunction(ϕ, α, Mstar0, mmin, mmax; npoints=1000)
    SchechterMassFunction(s::SchechterMassFunction, mmin, mmax; npoints=1000)

These constructors support random sampling from the returned instance `s` via `rand(s)`, `rand(s, 1000)` and so on. For speed a look-up table of the inverse CDF is used for sampling; coverage of this look-up table is defined by the keyword arguments `mmin` and `mmax` which give the limits of the look-up table in solar masses. The number of elements in this look-up table is `npoints`; more points gives greater sampling accuracy at the cost of increased memory and decreased sampling speed. The default `npoints=1000` is typically sufficient.

```jldoctest
julia> s = SchechterMassFunction(1.0, 1.2, 1e6) # Basic instance
SchechterMassFunction{Float64, Nothing}(1.0, 1.2, 1.0e6, nothing)

julia> s(1e6) isa Float64 # call to evaluate mass function
true

julia> s2 = SchechterMassFunction(s, 1e6, 1e11; npoints=1000); # instance with inverse CDF cache

julia> rand(s2) isa Float64 # s2 supports random sampling
true
```
"""
struct SchechterMassFunction{T,S} <: ConstantMassFunction{T}
    ϕ::T
    α::T
    Mstar0::T
    icdf::S
end
function SchechterMassFunction(ϕ, α, Mstar0)
    T = promote_type(typeof(ϕ), typeof(α), typeof(Mstar0))
    return SchechterMassFunction{T,Nothing}(T(ϕ), T(α), T(Mstar0), nothing)
end
SchechterMassFunction(ϕ, α, Mstar0, mmin, mmax; npoints::Int=1000) = SchechterMassFunction(SchechterMassFunction(ϕ, α, Mstar0), mmin, mmax; npoints)
function SchechterMassFunction(s::SchechterMassFunction{T}, mmin, mmax; npoints::Int=1000) where T
    x = logrange(mmin, mmax, npoints)
    y = s.(x)
    # logx = log10.(x)
    icdf = inverse_cdf!(y, x)
    return SchechterMassFunction{T,typeof(icdf)}(s.ϕ, s.α, s.Mstar0, icdf)
end

function (s::SchechterMassFunction)(Mstar)
    mass_ratio = Mstar / s.Mstar0
    return logten * s.ϕ * (mass_ratio)^(s.α + 1) * exp(-mass_ratio)
end
# Random sampling takes ~2x longer than it should, but not worried for now
# icdf(s::SchechterMassFunction, x) = s.icdf(x)
# _rand(s::SchechterMassFunction, u) = s.icdf(u)
# @inline Random.rand(rng::Random.AbstractRNG, s::SchechterMassFunction) = s.icdf(rand(rng))
# function Random.rand(rng::Random.AbstractRNG, s::SchechterMassFunction, dims::Dims)
#     return reshape([rand(rng, s) for _ in 1:prod(dims)], dims)
# end
# function Random.rand(rng::Random.AbstractRNG, s::SchechterMassFunction{T,Nothing}) where T
#     error("Provided SchechterMassFunction has no inverse CDF buffer; please create new instance with `mmin` and `mmax` arguments.")
# end

"""
    DoubleSchechterMassFunction(ϕ1, α1, Mstar0_1, ϕ2, α2, Mstar0_2)
A double Schechter mass function model, essentially just the sum of two Schechter models with different parameters. Let ``S(\\phi_*, \\alpha, M_{*,0}, M_*)`` be the single Schechter mass function [`SchechterMassFunction`](@ref), then this function is simply ``S(\\phi_{*,1}, \\alpha_1, M_{*,0_1}, M_*) + S(\\phi_{*,2}, \\alpha_2, M_{*,0_2}, M_*)``.

Calling an instance of `DoubleSchechterMassFunction` with a stellar mass (in solar masses) will return the value of the mass function.

    DoubleSchechterMassFunction(ϕ1, α1, Mstar0_1, ϕ2, α2, Mstar0_2, mmin, mmax; npoints=1000)
    DoubleSchechterMassFunction(s::DoubleSchechterMassFunction, mmin, mmax; npoints=1000)
These constructors support random sampling from the returned instance `s` via `rand(s)`, `rand(s, 1000)` and so on. For speed a look-up table of the inverse CDF is used for sampling; coverage of this look-up table is defined by the keyword arguments `mmin` and `mmax` which give the limits of the look-up table in solar masses. The number of elements in this look-up table is `npoints`; more points gives greater sampling accuracy at the cost of increased memory and decreased sampling speed. The default `npoints=1000` is typically sufficient.

```jldoctest
julia> s = DoubleSchechterMassFunction(8.9e-4, -1.4, 1e11, 8.31e-5, 0.5, exp10(10.64))
DoubleSchechterMassFunction{Float64, Nothing}(SchechterMassFunction{Float64, Nothing}(0.00089, -1.4, 1.0e11, nothing), SchechterMassFunction{Float64, Nothing}(8.31e-5, 0.5, 4.3651583224016655e10, nothing), nothing)

julia> s(1e6) isa Float64 # Call to evaluate mass function
true

julia> s2 = DoubleSchechterMassFunction(s, 1e6, 1e11; npoints=1000); # instance with inverse CDF cache

julia> rand(s2) isa Float64 # s2 supports random sampling
true
```
"""
struct DoubleSchechterMassFunction{T,S} <: ConstantMassFunction{T}
    S1::SchechterMassFunction{T}
    S2::SchechterMassFunction{T}
    icdf::S
end
function DoubleSchechterMassFunction(ϕ1, α1, Mstar0_1, ϕ2, α2, Mstar0_2)
    T = promote_type(typeof(ϕ1), typeof(α1), typeof(Mstar0_1),
                     typeof(ϕ2), typeof(α2), typeof(Mstar0_2))
    S1 = SchechterMassFunction(ϕ1, α1, Mstar0_1)
    S2 = SchechterMassFunction(ϕ2, α2, Mstar0_2)
    return DoubleSchechterMassFunction{T,Nothing}(S1, S2, nothing)
end
DoubleSchechterMassFunction(ϕ1, α1, Mstar0_1, ϕ2, α2, Mstar0_2, mmin, mmax; npoints::Int=1000) = DoubleSchechterMassFunction(DoubleSchechterMassFunction(ϕ1, α1, Mstar0_1, ϕ2, α2, Mstar0_2), mmin, mmax; npoints)
function DoubleSchechterMassFunction(s::DoubleSchechterMassFunction{T}, mmin, mmax; npoints::Int=1000) where T
    x = logrange(mmin, mmax, npoints)
    y = s.(x)
    icdf = inverse_cdf!(y, x)
    return DoubleSchechterMassFunction{T,typeof(icdf)}(s.S1, s.S2, icdf)
end
(s::DoubleSchechterMassFunction)(Mstar) = s.S1(Mstar) + s.S2(Mstar)