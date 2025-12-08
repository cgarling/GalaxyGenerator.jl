module GalaxyGeneratorUnitfulExt # Same name as file

import GalaxyGenerator as GG
import Unitful as u

for T in (GG.EGG.NoIGM, GG.EGG.Madau1995IGM)
    @eval GG.EGG.transmission(s::$T, z, λ_r::u.Length)= GG.EGG.transmission(s, z, u.ustrip(u.angstrom, λ_r))
    @eval GG.EGG.tau(s::$T, z, λ_r::u.Length)= GG.EGG.tau(s, z, u.ustrip(u.angstrom, λ_r))
end

end # module