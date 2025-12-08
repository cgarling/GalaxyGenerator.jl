# [IGM](@id IGM)

Here we describe available models for calculating the attenuation due to IGM absorption.

# Methods
```@docs
GalaxyGenerator.IGM.tau
GalaxyGenerator.IGM.transmission
```

# Models
```@docs
GalaxyGenerator.IGM.NoIGM
GalaxyGenerator.IGM.Madau1995IGM
GalaxyGenerator.IGM.Inoue2014IGM
```

# Model Comparison
Below we use Makie.jl to plot a comparison between the different IGM attenuation model. This can be compared to, e.g., Figure 4 of [Inoue2014](@citet).

```@example plotting
using CairoMakie
using GalaxyGenerator.IGM
models = [Madau1995IGM(), Inoue2014IGM()]
ls = [:dash, :solid]
labels = ["Madau1995IGM", "Inoue2014IGM"]
# λ_o = 0.3:0.01:1.0 # 0.3 to 1.0 μm observer-frame wavelengths
λ_o = 3000:10:10000 # 3k to 10k angstroms; 0.3 to 1.0 μm observer-frame wavelengths
z  = [2, 3, 4, 5, 6]

fig = Figure()
ax = Axis(fig[1, 1], xlabel=L"$\lambda$ [Å]", ylabel="Transmission")
for i in eachindex(z)
    λ_r = λ_o ./ (1 + z[i])
    for j in eachindex(models)
        trans = transmission.(models[j], z[i], λ_r)
        lines!(ax, λ_o, trans, color = :black, linestyle=ls[j]) # label=labels[j],
        text!(ax, 3900, 0.97; text=L"z_S=2", align=(:left, :top), space = :data)
        text!(ax, 5100, 0.97; text=L"z_S=3", align=(:left, :top), space = :data)
        text!(ax, 6200, 0.97; text=L"z_S=4", align=(:left, :top), space = :data)
        text!(ax, 7400, 0.97; text=L"z_S=5", align=(:left, :top), space = :data)
        text!(ax, 8700, 0.97; text=L"z_S=6", align=(:left, :top), space = :data)
    end
end
ax.xticks = collect(range(3000, 9000; step=1000))
xlims!(ax, minimum(λ_o), maximum(λ_o))

legend_elements = [LineElement(color = :black, linestyle = ls[i], label = labels[i]) for i in eachindex(models)]
legend = Legend(fig[1, 1], legend_elements, labels, orientation = :vertical,
                tellheight=false, tellwidth=false, halign=:right, valign=:center)
fig
```

## IGM References
This page cites the following references:

```@bibliography
Pages = ["IGM.md"]
Canonical = false
```
