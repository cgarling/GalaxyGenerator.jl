module GalaxyGenerator
using ArgCheck: @argcheck, @check
using Compat: logrange
using DataInterpolations: AkimaInterpolation
using Distributions: LogNormal, Normal, Uniform, mean
using IrrationalConstants: logten
using Random: Random, default_rng, AbstractRNG
using Trapz: trapz
using SpecialFunctions: erf

export SchechterMassFunction, DoubleSchechterMassFunction

include("random.jl")
include("Schechter.jl")
include("EGG.jl")

end
