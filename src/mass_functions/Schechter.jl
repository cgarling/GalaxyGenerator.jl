# Code for Schechter luminosity/mass functions

"""
    SchechterMassFunction(ϕ, α, Mstar0)
The Schechter mass function model, defined as 

```math
S = \\text{log}(10) \\ \\phi_* \\left( \\frac{M_*}{M_{*,0}} \\right)^{\\alpha+1} \\text{exp} \\left(- \\frac{M_*}{M_{*,0}} \\right)
```

Calling an instance of `SchechterMassFunction` with a stellar mass (in solar masses) will return the value of the mass function. 

```jldoctest schechter
julia> using GalaxyGenerator: SchechterMassFunction

julia> s = SchechterMassFunction(1.0, 1.2, 1e6)
SchechterMassFunction{Float64}(1.0, 1.2, 1.0e6)

julia> s(1e6) isa Float64 # call to evaluate mass function
true
```

Random samples can be drawn by creating a [`MassFunctionSampler`](@ref) instance:

```jldoctest schechter
julia> using GalaxyGenerator.MassFunctions: SchechterMassFunction, MassFunctionSampler

julia> s2 = MassFunctionSampler(s, 1e6, 1e11; npoints=1000);

julia> rand(s2) isa Float64
true
```
"""
struct SchechterMassFunction{T} <: ConstantMassFunction{T}
    ϕ::T
    α::T
    Mstar0::T
    function SchechterMassFunction(ϕ, α, Mstar0)
        T = promote_type(typeof(ϕ), typeof(α), typeof(Mstar0))
        new{T}(ϕ, α, Mstar0)
    end
end

function (s::SchechterMassFunction)(Mstar)
    mass_ratio = Mstar / s.Mstar0
    return logten * s.ϕ * (mass_ratio)^(s.α + 1) * exp(-mass_ratio)
end


"""
    DoubleSchechterMassFunction(ϕ1, α1, Mstar0_1, ϕ2, α2, Mstar0_2)
A double Schechter mass function model, essentially just the sum of two Schechter models with different parameters. Let ``S(\\phi_*, \\alpha, M_{*,0}, M_*)`` be the single Schechter mass function [`SchechterMassFunction`](@ref), then this function is simply ``S(\\phi_{*,1}, \\alpha_1, M_{*,0_1}, M_*) + S(\\phi_{*,2}, \\alpha_2, M_{*,0_2}, M_*)``.

Calling an instance of `DoubleSchechterMassFunction` with a stellar mass (in solar masses) will return the value of the mass function.

```jldoctest double_schechter
julia> using GalaxyGenerator: DoubleSchechterMassFunction

julia> s = DoubleSchechterMassFunction(8.9e-4, -1.4, 1e11, 8.31e-5, 0.5, exp10(10.64))
DoubleSchechterMassFunction{Float64}(SchechterMassFunction{Float64}(0.00089, -1.4, 1.0e11), SchechterMassFunction{Float64}(8.31e-5, 0.5, 4.3651583224016655e10))

julia> s(1e6) isa Float64 # Call to evaluate mass function
true
```

Random samples can be drawn by creating a [`MassFunctionSampler`](@ref) instance:

```jldoctest double_schechter
julia> using GalaxyGenerator.MassFunctions: MassFunctionSampler

julia> s2 = MassFunctionSampler(s, 1e6, 1e11; npoints=1000);

julia> rand(s2) isa Float64
true
```
"""
struct DoubleSchechterMassFunction{T} <: ConstantMassFunction{T}
    S1::SchechterMassFunction{T}
    S2::SchechterMassFunction{T}
end

function DoubleSchechterMassFunction(ϕ1, α1, Mstar0_1, ϕ2, α2, Mstar0_2)
    T = promote_type(typeof(ϕ1), typeof(α1), typeof(Mstar0_1),
                     typeof(ϕ2), typeof(α2), typeof(Mstar0_2))
    S1 = SchechterMassFunction(ϕ1, α1, Mstar0_1)
    S2 = SchechterMassFunction(ϕ2, α2, Mstar0_2)
    return DoubleSchechterMassFunction{T}(S1, S2)
end

(s::DoubleSchechterMassFunction)(Mstar) = s.S1(Mstar) + s.S2(Mstar)