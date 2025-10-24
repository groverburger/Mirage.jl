using Documenter, Mirage

makedocs(
    sitename = "Mirage.jl Documentation",
    modules = [Mirage],
    pages = [
        "Home" => "index.md",
    ],
    checkdocs = :all,
    warnonly = [:missing_docs],  # Only warn, don't fail
    remotes = nothing
)
