# Getting Started

This guide will walk you through installing Mirage.jl, setting up a basic scene, and rendering your first frame.

## Installation

First, you need to have Julia installed on your system. You can download it from the [official Julia website](https://julialang.org/downloads/).

Once Julia is installed, open the Julia REPL (interactive terminal) and use the built-in package manager, `Pkg`, to add Mirage.jl.

```julia
import Pkg
Pkg.add("Mirage")
```

## Your First Render: A Simple Rectangle

The core of a Mirage.jl application consists of two parts: initialization and a render loop.

1.  `initialize()`: Sets up the window, graphics context, and all necessary resources.
2.  `start_render_loop(render_function)`: Starts a loop that repeatedly calls your custom `render_function` to draw each frame.

Let's create a simple application that draws a red rectangle on a black background.

```julia
import Mirage

# The main function of our application
function main()
    # 1. Initialize the window and rendering context.
    # This creates an 800x600 window by default.
    Mirage.initialize()
    
    # 2. Define the function that will be called for every frame.
    function render()
        # Clear the screen to a black color.
        Mirage.clear()
        
        # Set the drawing color to red.
        # `rgba` is a helper to create a color from Red, Green, Blue, and Alpha values (0-255).
        Mirage.fillcolor(Mirage.rgba(255, 0, 0))
        
        # Draw a filled rectangle.
        # The arguments are x, y, width, height.
        Mirage.fillrect(100, 100, 200, 150)
    end
    
    # 3. Start the render loop, passing our `render` function.
    # This will run until the user closes the window.
    Mirage.start_render_loop(render)
end

# Run the main function
main()
```

### Running the Code

Save the code above as a Julia file (e.g., `my_app.jl`) and run it from your terminal:

```sh
julia my_app.jl
```

You should see a window appear with a red rectangle.

## Common Troubleshooting

*   **Window doesn't appear or closes immediately:**
    *   Ensure your graphics drivers are up to date.
    *   Make sure you are calling `start_render_loop()`. Without it, the program will initialize, draw one frame, and then exit.
*   **"Mirage not found" error:**
    *   Ensure you have installed the package correctly using `Pkg.add("Mirage")`.
    *   If you are running the code from a script, make sure `using Mirage` is at the top.
*   **Errors related to file loading (e.g., textures, models):**
    *   File paths are relative to where you run the Julia script. Ensure the files are in the correct location. For example, if your script needs `assets/player.png`, that path should exist relative to your terminal's current directory.
