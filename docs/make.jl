using UnnormalizedModels
using Documenter

DocMeta.setdocmeta!(UnnormalizedModels, :DocTestSetup, :(using UnnormalizedModels); recursive=true)

makedocs(;
    modules=[UnnormalizedModels],
    authors="aconitum3 <aconitum3@example.com> and contributors",
    sitename="UnnormalizedModels.jl",
    format=Documenter.HTML(;
        canonical="https://aconitum3.github.io/UnnormalizedModels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/aconitum3/UnnormalizedModels.jl",
    devbranch="main",
)
