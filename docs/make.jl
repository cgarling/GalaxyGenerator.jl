using GalaxyGenerator
using Documenter

DocMeta.setdocmeta!(GalaxyGenerator, :DocTestSetup, :(using GalaxyGenerator); recursive=true)

makedocs(;
    modules=[GalaxyGenerator],
    authors="cgarling <chris.t.garling@gmail.com> and contributors",
    sitename="GalaxyGenerator.jl",
    format=Documenter.HTML(;
        canonical="https://cgarling.github.io/GalaxyGenerator.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
    doctest=false,
    linkcheck=true,
    warnonly=[:missing_docs, :linkcheck],
)

deploydocs(;
    repo="github.com/cgarling/GalaxyGenerator.jl",
    devbranch="main",
)
