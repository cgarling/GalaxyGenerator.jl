"""
    BinnedRedshiftMassFunction(redshifts::AbstractVector, mass_functions::AbstractVector{<:ConstantMassFunction}, extrapolate::Bool)
A redshift-dependent mass function that switches between different `ConstantMassFunction` instances at specified redshifts. For intermediate redshifts, the mass function is linearly interpolated between the nearest redshifts.

# Fields
- `redshifts`: A sorted vector of redshifts defining the range of validity for each mass function.
- `mass_functions`: A vector of `ConstantMassFunction` instances corresponding to the redshifts.
- `extrapolate`: A boolean flag indicating whether to extrapolate beyond the valid redshift range.

# Usage
```jldoctest
julia> using GalaxyGenerator.MassFunctions: BinnedRedshiftMassFunction, SchechterMassFunction

julia> redshifts = [0.0, 1.0, 2.0];

julia> mfs = [SchechterMassFunction(1.0, -1.3, 1e11),
             SchechterMassFunction(0.8, -1.5, 1e11),
             SchechterMassFunction(0.5, -1.7, 1e11)];

julia> binned = BinnedRedshiftMassFunction(redshifts, mfs, true);

julia> binned(1e10, 0.0) == mfs[1](1e10) # At z=0.0
true

julia> binned(1e10, 2.0) == mfs[3](1e10) # At z=2.0
true

julia> isapprox(binned(1e10, 0.5), (mfs[2](1e10) + mfs[1](1e10)) / 2) # Interpolates between z=0.0 and z=1.0
true

julia> binned(1e10, 3.0) == binned(1e10, last(redshifts)) # Extrapolates using the value at max(z)
true
```
"""
struct BinnedRedshiftMassFunction{T,A,B} <: RedshiftMassFunction{T}
    redshifts::A
    mass_functions::B
    extrapolate::Bool

    function BinnedRedshiftMassFunction(redshifts::AbstractVector{T}, mass_functions::AbstractVector{<:ConstantMassFunction}, extrapolate::Bool) where T
        @argcheck issorted(redshifts) "Redshifts must be sorted in ascending order."
        @argcheck length(redshifts) == length(mass_functions) "Number of redshifts must match number of mass functions."
        U = promote_type(T, typeof(first(mass_functions)(1e9)))
        return new{U, typeof(redshifts), typeof(mass_functions)}(redshifts, mass_functions, extrapolate)
    end
end

function (binned::BinnedRedshiftMassFunction)(Mstar, z)
    redshifts = binned.redshifts
    mass_functions = binned.mass_functions

    if z < redshifts[1]
        if binned.extrapolate
            return mass_functions[1](Mstar)
        else
            throw(DomainError("Redshift z is below the valid range."))
        end
    elseif z > redshifts[end]
        if binned.extrapolate
            return mass_functions[end](Mstar)
        else
            throw(DomainError("Redshift z is above the valid range."))
        end
    elseif z == redshifts[1]
        return mass_functions[1](Mstar)
    elseif z == redshifts[end]
        return mass_functions[end](Mstar)
    end

    # Find the nearest redshifts for interpolation
    i = searchsortedlast(redshifts, z)
    z1, z2 = redshifts[i], redshifts[i+1]
    mf1, mf2 = mass_functions[i], mass_functions[i+1]

    # Interpolate linearly between the two mass functions
    value1 = mf1(Mstar)
    value2 = mf2(Mstar)
    weight = (z - z1) / (z2 - z1)  # Compute interpolation weight
    return value1 * (1 - weight) + value2 * weight
end