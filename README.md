# Mirage.jl

[![built with julia](https://julialang.org/assets/infra/built-with-julia-logo-dark.svg)](https://julialang.org)

Mirage.jl is a hardware-accelerated 2D and 3D graphics library for Julia. It provides a simple, user-friendly API for creating interactive graphics applications, games, and data visualizations. By leveraging the power of OpenGL, Mirage.jl delivers high performance for real-time rendering tasks.

## Features

- **2D and 3D Rendering:** Seamlessly switch between 2D and 3D rendering modes.
- **OpenGL Backend:** High-performance rendering powered by ModernGL.jl.
- **Immediate Mode API:** A simple and intuitive API for drawing shapes, text, and images.
- **Transformations:** Easy-to-use functions for translation, rotation, and scaling.
- **Shaders:** Support for custom shaders to create advanced visual effects.
- **Text Rendering:** Built-in text rendering capabilities.
- **Image Loading:** Load and display images in various formats.

## Getting Started

To get started with Mirage.jl, you first need to have Julia installed on your system. You can download it from the [official Julia website](https://julialang.org/downloads/).

Once you have Julia installed, you can add Mirage.jl as a dependency to your project.

First, start the Julia REPL by typing `julia` in your terminal. Then, press `]` to enter the package manager.

```julia
(@v1.10) pkg> add Mirage
```

## Usage

Here is a simple example of how to create a window and draw a rotating square:

```julia
using Mirage

function main()
    initialize(window_width=800, window_height=600)

    start_render_loop(function()
        # Clear the screen
        clear()

        # Save the current state
        save()

        # Translate to the center of the screen
        translate(400, 300)

        # Rotate the square
        rotate(time())

        # Set the fill color to red
        fillcolor(rgba(255, 0, 0))

        # Draw a rectangle
        rect(-50, -50, 100, 100)

        # Fill the rectangle
        fill()

        # Restore the state
        restore()
    end)
end

main()
```

## API

The Mirage.jl API is designed to be simple and intuitive. Here are some of the main functions:

- `initialize()`: Initializes the rendering context and creates a window.
- `start_render_loop(render_function)`: Starts the main render loop.
- `clear()`: Clears the screen.
- `save()` and `restore()`: Save and restore the current drawing state.
- `translate(x, y, z)`: Translates the current drawing context.
- `rotate(angle)`: Rotates the current drawing context.
- `scale(x, y, z)`: Scales the current drawing context.
- `beginpath()`, `moveto(x, y)`, `lineto(x, y)`, `closepath()`: Create paths.
- `fill()`, `stroke()`: Fill and stroke paths.
- `fillcolor(color)`, `strokecolor(color)`, `strokewidth(width)`: Set drawing styles.
- `rect(x, y, w, h)`, `circle(r, x, y)`: Draw shapes.
- `text(string)`: Draw text.
- `drawimage(x, y, w, h, texture)`: Draw an image.

## Contributing

Contributions to Mirage.jl are welcome! If you find a bug or have a feature request, please open an issue on the [GitHub repository](https://github.com/your-username/Mirage.jl). If you would like to contribute code, please fork the repository and submit a pull request.
