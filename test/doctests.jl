using GalaxyGenerator
using Documenter: DocMeta, doctest

DocMeta.setdocmeta!(GalaxyGenerator, :DocTestSetup, :(using GalaxyGenerator); recursive=true)
doctest(GalaxyGenerator)
