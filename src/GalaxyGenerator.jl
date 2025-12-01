module GalaxyGenerator
using ArgCheck: @argcheck, @check
using Compat: logrange
using DataInterpolations: AkimaInterpolation
using IrrationalConstants: logten
import Random
using Trapz: trapz

export SchechterMassFunction, DoubleSchechterMassFunction

include("random.jl")
include("Schechter.jl")

end
