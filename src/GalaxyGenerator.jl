module GalaxyGenerator

using ArgCheck: @argcheck, @check
using Compat: logrange
using DataInterpolations: AkimaInterpolation
using IrrationalConstants: logten
using Random: Random, default_rng, AbstractRNG

export SchechterMassFunction, DoubleSchechterMassFunction

include("utils.jl")
include("IGM/IGM.jl")
using .IGM
include("EmissionLines.jl")
include("Schechter.jl")
include("EGG/EGG.jl")
using .EGG

end
