# Core Concepts

Mirage.jl is built around a few core concepts designed to provide a flexible and intuitive environment for real-time graphics. Understanding these will help you get the most out of the library.

## Immediate-Mode Rendering

Mirage.jl uses an **immediate-mode** rendering paradigm. This means that graphics commands (like `fillrect()` or `draw_mesh()`) are executed and sent to the GPU for rendering right away. The library does not maintain a persistent scene graph of objects to be drawn. Instead, **you are responsible for drawing everything you want to see on the screen in every single frame.**

This approach is highly flexible and gives you full control over the rendering process. It is often simpler to understand and debug than retained-mode systems, especially for 2D graphics, data visualization, and prototyping.

### Comparison to HTML5 Canvas

The 2D API of Mirage.jl is heavily inspired by the **HTML5 Canvas API**. If you have ever used `canvas.getContext('2d')` in JavaScript, you will find the concepts familiar:

-   Drawing commands are issued sequentially.
-   A state machine tracks transformations, colors, and styles.
-   `save()` and `restore()` are used to manage this state.

## The State Stack: `save()` and `restore()`

One of the most powerful features of the 2D API is the graphics state stack. The current state includes:

-   The current transformation matrix (position, rotation, and scale).
-   Fill and stroke colors.
-   Stroke width.

When you call `save()`, Mirage.jl pushes a complete copy of the current graphics state onto a stack. When you call `restore()`, it pops the last-saved state off the stack, instantly reverting all changes made since the last `save()`.

This is extremely useful for hierarchical drawing. For example, to draw a planet and its moon:

1.  `save()` the current state.
2.  `translate()` to the planet's position.
3.  Draw the planet.
4.  `save()` the planet-relative state.
5.  `translate()` to the moon's position relative to the planet.
6.  `rotate()` the moon around its own axis.
7.  Draw the moon.
8.  `restore()` to return to the planet-relative state (before the moon was drawn).
9.  `restore()` again to return to the original state (before the planet was drawn).

```julia
# Pseudocode
save()       # Save world origin state
translate(300, 300) # Move to planet's position
draw_planet()

save()       # Save planet state
translate(100, 0)   # Move to moon's position relative to planet
rotate(angle)  # Rotate the moon
draw_moon()

restore()    # Restore to planet state
restore()    # Restore to world origin state
```

## The Rendering Pipeline: Meshes and Shaders

Behind the scenes, all drawing operations in Mirage.jl are ultimately converted into `Mesh` objects and drawn to the screen.

-   **`Mesh`**: A `Mesh` is a container for vertex data stored on the GPU. This includes vertex positions, texture coordinates, normals, etc. Mirage provides helpers like `create_cube()` and `create_uv_sphere()` to generate primitive meshes, and `load_obj_mesh()` to load them from files.

-   **`ShaderInfo`**: This holds a compiled GLSL shader program. Mirage.jl uses a default shader for all its 2D and 3D drawing, which handles basic position, color, and texture mapping.

-   **`draw_mesh(mesh, texture, color)`**: This is the fundamental drawing call. It tells the GPU to draw the vertex data from a `Mesh` using the active shader, applying a given texture and tint color.

When you call a 2D function like `fillrect()`, Mirage.jl performs these steps internally:

1.  Gets a pre-defined "immediate mode" mesh (a simple quad).
2.  Updates its vertices to match the rectangle's position and size.
3.  Calls `draw_mesh()` with a blank white texture, tinted by the current fill color.
