module GalaxyGenerator

using ArgCheck: @argcheck, @check

export SchechterMassFunction, DoubleSchechterMassFunction

include("utils.jl")
include("IGM/IGM.jl")
using .IGM
include("EmissionLines.jl")
include("mass_functions/mass_functions.jl")
using .MassFunctions
include("EGG/EGG.jl")
using .EGG

end
