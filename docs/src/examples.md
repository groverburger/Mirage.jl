# Examples

This page contains extended examples and real-world use cases for Mirage.jl.

## 2D Drawing Example

Mirage.jl provides a stateful 2D drawing API similar to the HTML5 Canvas. You can apply transformations (translate, rotate, scale), set colors, and draw shapes. The state is managed on a stack using `save()` and `restore()`.

This example demonstrates basic shape drawing, transformations, and text rendering.

```julia
using Mirage

function main()
    initialize(window_width=1280, window_height=720)

    # Load a texture for the sprite
    # Replace with a path to your own image
    test_texture = load_texture("test_texture.png")

    angle = 0.0

    function render()
        clear() # Clear the screen
        update_ortho_projection_matrix() # Use a 2D orthographic camera

        # --- Draw a rotating, textured sprite ---
        save() # Save the current state (default transform)
        fillcolor(rgba(255, 255, 255)) # White tint
        
        # Move to the center of the screen
        translate(640, 360)
        # Rotate over time
        rotate(angle)
        # Scale up
        scale(4.0)
        # Draw the image centered at the new origin
        drawimage(-50, -50, 100, 100, test_texture)
        
        restore() # Restore to the default state

        # --- Draw a path with stroke and fill ---
        save()
        beginpath()
        strokewidth(5)
        strokecolor(rgba(255, 255, 0, 255)) # Yellow stroke
        moveto(400, 400)
        lineto(500, 400)
        lineto(500, 500)
        lineto(400, 500)
        closepath() # Connects the last point to the first
        
        fillcolor(rgba(0, 0, 255, 100)) # Semi-transparent blue fill
        fill()    # Fill the path
        stroke()  # Stroke the path
        restore()

        # --- Draw Text ---
        save()
        translate(50, 650)
        scale(2)
        fillcolor(rgba(20, 200, 255, 255)) # Cyan color
        text("Mirage.jl 2D Demo")
        restore()

        # Update animation variable
        angle += 0.01
    end

    start_render_loop(render)
end

main()
```

## 3D Scene Example

For 3D, you switch to a perspective camera and work with `Mesh` objects. A `Mesh` holds vertex data on the GPU and can be created with primitive shapes or loaded from files.

This example shows a camera rotating around a 3D cube.

```julia
using Mirage

function main()
    initialize()

    # Create a 3D cube mesh with a side length of 10 units
    cube_mesh = create_cube(10.0)
    
    # Enable the depth test for correct 3D rendering
    glEnable(GL_DEPTH_TEST)

    frame = 0

    function render()
        clear()
        
        # Use a perspective camera for 3D
        update_perspective_projection_matrix()

        # Set up the camera position and orientation.
        # The camera will circle around the origin (0,0,0).
        camera_x = cos(frame / 100) * 30
        camera_y = sin(frame / 100) * 30
        lookat(
            Float32[camera_x, camera_y, 20], # Camera position
            Float32[0, 0, 0],                 # Target to look at
            Float32[0, 0, 1]                  # Up direction (Z-up)
        )

        # --- Draw the cube ---
        save()
        # Apply a rotation to the cube itself
        rotate(frame / 50, frame / 30, frame / 20)
        fillcolor(rgba(0, 150, 255)) # Blue tint
        draw_mesh(cube_mesh)
        restore()

        frame += 1
    end

    start_render_loop(render)
end

main()
```

## Loading a 3D Model (.obj)

Mirage can load 3D models from Wavefront `.obj` files using the `load_obj_mesh` function. The model must be in the same directory as your script or you must provide a correct relative path.

```julia
using Mirage

function main()
    initialize()

    # Load the mesh from an OBJ file
    # This requires a 'cube.obj' file in the same directory
    try
        obj_mesh = load_obj_mesh("cube.obj")
        
        glEnable(GL_DEPTH_TEST)
        frame = 0

        function render()
            clear()
            update_perspective_projection_matrix()

            # Position the camera
            lookat(Float32[30, 30, 30], Float32[0, 0, 0], Float32[0, 0, 1])

            # Draw the loaded mesh
            save()
            rotate(0.0, 0.0, frame / 100) # Rotate around Z-axis
            scale(5.0) # Make it 5 times bigger
            fillcolor(rgba(255, 255, 255))
            draw_mesh(obj_mesh)
            restore()

            frame += 1
        end

        start_render_loop(render)

    catch e
        @error "Could not load 'cube.obj'. Make sure the file exists." e
    end
end

main()
```
