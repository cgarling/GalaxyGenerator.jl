module GalaxyGenerator

using ArgCheck: @argcheck, @check

export SchechterMassFunction, DoubleSchechterMassFunction, MassFunctionSampler, generate_galaxies, egg

include("utils.jl")
include("IGM/IGM.jl")
using .IGM
include("EmissionLines.jl")
include("mass_functions/mass_functions.jl")
using .MassFunctions
include("EGG/EGG.jl")
using .EGG

end
