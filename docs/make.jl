using GalaxyGenerator
using Documenter
using DocumenterCitations: CitationBibliography

using CairoMakie
# Makie plotting options
set_theme!(theme_latexfonts(); fontsize = 30, size = (800, 800), 
    Axis = (xminorticksvisible=true, yminorticksvisible=true, xminorticks = IntervalsBetween(5), yminorticks = IntervalsBetween(5), xticksize=10.0, yticksize=10.0)
)

# Check if on CI
const CI = get(ENV, "CI", nothing) == "true"

DocMeta.setdocmeta!(GalaxyGenerator, :DocTestSetup, :(using GalaxyGenerator); recursive=true)
bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :authoryear)

makedocs(;
    sitename="GalaxyGenerator.jl",
    format = Documenter.HTML(prettyurls = CI,
        canonical="https://cgarling.github.io/GalaxyGenerator.jl",
        edit_link="main",
        assets = String["assets/citations.css"],
        size_threshold_warn = 409600, # v1.0.0 default: 102400 (bytes)
        size_threshold = 819200,      # v1.0.0 default: 204800 (bytes)
        example_size_threshold=0),    # Write all @example to file
    modules = [GalaxyGenerator],
    authors = "Chris Garling",
    pages = [
        "Home" => "index.md",
        "EGG.md",
        "IGM.md",
        "refs.md",
    ],
    doctest = false,
    linkcheck = CI,
    warnonly = [:missing_docs, :linkcheck],
    plugins = [bib]
)

deploydocs(;
    repo="github.com/cgarling/GalaxyGenerator.jl",
    devbranch="main",
)
