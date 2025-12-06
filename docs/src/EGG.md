# [Empirical Galaxy Generator (EGG)](@id EGG)

The Empirical Galaxy Generator (EGG) was developed by [Schreiber2017](@citet) and the C++ implementation was released as open-source [here](https://github.com/cschreib/egg) under the MIT license.

In short, EGG builds galaxy catalogs by leveraging empirical scaling relations between different galaxy properties. The "root properties" for EGG are the galaxy stellar mass, redshift, and whether the galaxy is actively star-forming or quiescent. All other properties are derived from this starting point.

# Stellar Mass Functions
To make predictions with EGG, we first need a realistic catalog of galaxy redshifts and stellar masses. The stellar mass functions originally used in [Schreiber2017](@citet) are shown in Figure 1 of that paper with parameters given in Table A.1 for star-forming galaxies and Table A.2 for quiescent galaxies. We implement these mass functions for easy use as

```@docs
GalaxyGenerator.EGG.EGGMassFunction_SF
GalaxyGenerator.EGG.EGGMassFunction_Q
```


## EGG References
This page cites the following references:

```@bibliography
Pages = ["EGG.md"]
Canonical = false
```
