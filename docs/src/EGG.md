# [Empirical Galaxy Generator (EGG)](@id EGG)

The Empirical Galaxy Generator (EGG) was developed by [Schreiber2017](@citet) and the C++ implementation was released as open-source [here](https://github.com/cschreib/egg) under the MIT license.

In short, EGG builds galaxy catalogs by leveraging empirical scaling relations between different galaxy properties. The "root properties" for EGG are the galaxy stellar mass, redshift, and whether the galaxy is actively star-forming or quiescent. All other properties are derived from this starting point.

# Stellar Mass Functions
To make predictions with EGG, we first need a realistic catalog of galaxy redshifts and stellar masses. The stellar mass functions originally used in [Schreiber2017](@citet) are shown in Figure 1 of that paper with parameters given in Table A.1 for star-forming galaxies and Table A.2 for quiescent galaxies. We implement these mass functions for easy use as

```@docs
GalaxyGenerator.EGG.EGGMassFunction_SF
GalaxyGenerator.EGG.EGGMassFunction_Q
```

These mass functions are plotted below.

```@example plotting
using CairoMakie
using GalaxyGenerator.EGG: EGGMassFunction_SF, EGGMassFunction_Q

# Define the range of log10(Mstar) and extract redshifts
logMstar = 8:0.1:12
# Ranges of redshifts defined by Schreiber2017
z_ranges = [0.3 0.7; 0.7 1.2; 1.2 1.8; 1.8 2.5; 2.5 3.5; 3.5 4.5]

# Create the figure
fig = Figure(size=(1000,800))

# Left panel: Star-forming mass function
ax1 = Axis(fig[1, 1],
    title = "Star-Forming",
    xlabel = L"\log_{10}(M_\star / M_\odot)",
    ylabel = L"\log_{10}(\Phi / \mathrm{Mpc}^{-3} \ \mathrm{dex}^{-1})",
    xscale = log10,
    yscale = log10
)
for i in axes(z_ranges,1)
    z = (z_ranges[i,1] + z_ranges[i,2]) / 2
    Φ = [EGGMassFunction_SF(exp10(M), z) for M in logMstar]
    lines!(ax1, exp10.(logMstar), Φ, label = L"%$(z_ranges[i,1]) < z < %$(z_ranges[i,2])")
end
axislegend(ax1; position=:lb)

# Right panel: Quiescent mass function
ax2 = Axis(fig[1, 2],
    title = "Quiescent",
    xlabel = L"\log_{10}(M_\star / M_\odot)",
    xscale = log10,
    yscale = log10,
    yticklabelsvisible = false  # Remove y-axis labels
)
for i in axes(z_ranges,1)
    z = (z_ranges[i,1] + z_ranges[i,2]) / 2
    Φ = [EGGMassFunction_Q(exp10(M), z) for M in logMstar]
    lines!(ax2, exp10.(logMstar), Φ, label = L"%$(z_ranges[i,1]) < z < %$(z_ranges[i,2])")
end
axislegend(ax2)

# Link y-axes of both panels
linkyaxes!(ax1, ax2)
ylims!(ax1, 1e-5, 1e-1)
xlims!(ax1, 1e8, 1e12)
xlims!(ax2, 1e8, 1e12)

fig
```


# SED Templates
SED templates from [Schreiber2017](@citet) are tabulated as a function of U-V and V-J colors. An SED template is chosen for each galaxy by finding the nearest valid template to its sampled colors. Below we show the SED template grid, where U-V vs V-J bins with a valid SED template are yellow.

```@example plotting
using GalaxyGenerator.EGG: optlib

# Create the figure
fig = Figure(size=(800,800))
ax = Axis(fig[1, 1],
    title = "Valid SED Templates",
    xlabel = "U-V",
    ylabel = "V-J")

# Plot the heatmap
# heatmap!(ax, optlib.buv, optlib.bvj, optlib.use; colormap = :viridis)
# heatmap!(ax, optlib.bvj, optlib.buv, transpose(optlib.use); colormap = :viridis, categorical=true)
hm = heatmap!(ax, optlib.bvj, optlib.buv, transpose(optlib.use); colormap = Categorical(:viridis))
# Plot sequence
vj = -1.0:0.1:2.5
uv = 0.65 .* vj .+ 0.45
lines!(ax, uv, vj, color=:red, linewidth=2)
Colorbar(fig[1,2], hm, size=40) # , tellheight=false, tellwidth=false)

# Display the figure
fig
```

## EGG References
This page cites the following references:

```@bibliography
Pages = ["EGG.md"]
Canonical = false
```
