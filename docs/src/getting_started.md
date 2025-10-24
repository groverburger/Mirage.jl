# Getting Started

This guide will walk you through installing Mirage.jl, setting up a basic scene, and rendering your first image.

## Installation

To install Mirage.jl, open the Julia REPL and run:

```julia
import Pkg
Pkg.add("Mirage")
```

## Your First Render

Here's a simple example to get you started. This code will create a window and draw a triangle.

```julia
using Mirage

function main()
    start_render_loop()
end

main()
```

## Common Troubleshooting

*   **Window doesn't appear:** Make sure your graphics drivers are up to date.
*   **"Mirage not found" error:** Ensure you have installed the package correctly as shown in the installation section.
