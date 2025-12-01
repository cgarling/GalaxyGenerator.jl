# Code for Schechter luminosity/mass functions

abstract type AbstractMassFunction{T} end

"""
    SchechterMassFunction(ϕ, α, Mstar0)
The Schechter mass function model, defined as 

```math
\\text{log}(10) \\ \\phi_* \\left( \\frac{M_*}{M_{*,0}} \\right)^{\\alpha+1} \\text{exp} \\left(- \\frac{M_*}{M_{*,0}} \\right)
```
"""
struct SchechterMassFunction{T}
    ϕ::T
    α::T
    Mstar0::T
end
function SchechterMassFunction(ϕ, α, Mstar0)
    T = promote_type(typeof(ϕ), typeof(α), typeof(Mstar0))
    return SchechterMassFunction{T}(T(ϕ), T(α), T(Mstar0))
end

function (s::SchechterMassFunction)(Mstar::Number)
    mass_ratio = Mstar/s.Mstar0
    return logten * s.ϕ * (mass_ratio)^(s.α + 1) * exp(-mass_ratio)
end

"""
    DoubleSchechterMassFunction(ϕ1, α1, Mstar0_1, ϕ2, α2, Mstar0_2)
A double Schechter mass function model, essentially just the sum of two Schechter models with different parameters. Let ``S(\\phi_*, \\alpha, M_{*,0}, M_*)`` be the single Schechter mass function [`SchechterMassFunction`](@ref), then this function is simply ``S(\\phi_{*,1}, \\alpha_1, M_{*,0_1}, M_*) + S(\\phi_{*,2}, \\alpha_2, M_{*,0_2}, M_*)``.
"""
struct DoubleSchechterMassFunction{T}
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
(s::DoubleSchechterMassFunction)(Mstar::Number) = s.S1(Mstar) + s.S2(Mstar)