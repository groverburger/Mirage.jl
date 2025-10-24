# Examples

This page contains extended examples and real-world use cases for Mirage.jl.

## 2D Drawing Example

Mirage.jl provides a stateful 2D drawing API similar to the HTML5 Canvas. You can apply transformations (translate, rotate, scale), set colors, and draw shapes. The state is managed on a stack using `save()` and `restore()`.

This example demonstrates basic shape drawing, transformations, and text rendering.

```julia
import Mirage

function main()
    Mirage.initialize(window_width=1280, window_height=720)
    
    # Load a texture for the sprite
    # Replace with a path to your own image
    test_texture = Mirage.load_texture("test_texture.png")
    
    angle = 0.0
    
    function render()
        Mirage.clear() # Clear the screen
        Mirage.update_ortho_projection_matrix() # Use a 2D orthographic camera
        
        # --- Draw a rotating, textured sprite ---
        Mirage.save() # Save the current state (default transform)
        Mirage.fillcolor(Mirage.rgba(255, 255, 255)) # White tint
        
        # Move to the center of the screen
        Mirage.translate(640, 360)
        # Rotate over time
        Mirage.rotate(angle)
        # Scale up
        Mirage.scale(4.0)
        # Draw the image centered at the new origin
        Mirage.drawimage(-50, -50, 100, 100, test_texture)
        
        Mirage.restore() # Restore to the default state
        
        # --- Draw a path with stroke and fill ---
        Mirage.save()
        Mirage.beginpath()
        Mirage.strokewidth(5)
        Mirage.strokecolor(Mirage.rgba(255, 255, 0, 255)) # Yellow stroke
        Mirage.moveto(400, 400)
        Mirage.lineto(500, 400)
        Mirage.lineto(500, 500)
        Mirage.lineto(400, 500)
        Mirage.closepath() # Connects the last point to the first
        
        Mirage.fillcolor(Mirage.rgba(0, 0, 255, 100)) # Semi-transparent blue fill
        Mirage.fill()    # Fill the path
        Mirage.stroke()  # Stroke the path
        Mirage.restore()
        
        # --- Draw Text ---
        Mirage.save()
        Mirage.translate(50, 650)
        Mirage.scale(2)
        Mirage.fillcolor(Mirage.rgba(20, 200, 255, 255)) # Cyan color
        Mirage.text("Mirage.jl 2D Demo")
        Mirage.restore()
        
        # Update animation variable
        angle += 0.01
    end
    
    Mirage.start_render_loop(render)
end

main()
```

## 3D Scene Example

For 3D, you switch to a perspective camera and work with `Mesh` objects. A `Mesh` holds vertex data on the GPU and can be created with primitive shapes or loaded from files.

This example shows a camera rotating around a 3D cube.

```julia
import Mirage

function main()
    Mirage.initialize()
    
    # Create a 3D cube mesh with a side length of 10 units
    cube_mesh = Mirage.create_cube(10.0)
    
    # Enable the depth test for correct 3D rendering
    glEnable(GL_DEPTH_TEST)
    
    frame = 0
    
    function render()
        Mirage.clear()
        
        # Use a perspective camera for 3D
        Mirage.update_perspective_projection_matrix()
        
        # Set up the camera position and orientation.
        # The camera will circle around the origin (0,0,0).
        camera_x = cos(frame / 100) * 30
        camera_y = sin(frame / 100) * 30
        Mirage.lookat(
            Float32[camera_x, camera_y, 20], # Camera position
            Float32[0, 0, 0],                 # Target to look at
            Float32[0, 0, 1]                  # Up direction (Z-up)
        )
        
        # --- Draw the cube ---
        Mirage.save()
        # Apply a rotation to the cube itself
        Mirage.rotate(frame / 50, frame / 30, frame / 20)
        Mirage.fillcolor(Mirage.rgba(0, 150, 255)) # Blue tint
        Mirage.draw_mesh(cube_mesh)
        Mirage.restore()
        
        frame += 1
    end
    
    Mirage.start_render_loop(render)
end

main()
```

## Loading a 3D Model (.obj)

Mirage can load 3D models from Wavefront `.obj` files using the `load_obj_mesh` function. The model must be in the same directory as your script or you must provide a correct relative path.

```julia
import Mirage

function main()
    Mirage.initialize()
    
    # Load the mesh from an OBJ file
    # This requires a 'cube.obj' file in the same directory
    try
        obj_mesh = Mirage.load_obj_mesh("cube.obj")
        
        glEnable(GL_DEPTH_TEST)
        frame = 0
        
        function render()
            Mirage.clear()
            Mirage.update_perspective_projection_matrix()
            
            # Position the camera
            Mirage.lookat(Float32[30, 30, 30], Float32[0, 0, 0], Float32[0, 0, 1])
            
            # Draw the loaded mesh
            Mirage.save()
            Mirage.rotate(0.0, 0.0, frame / 100) # Rotate around Z-axis
            Mirage.scale(5.0) # Make it 5 times bigger
            Mirage.fillcolor(Mirage.rgba(255, 255, 255))
            Mirage.draw_mesh(obj_mesh)
            Mirage.restore()
            
            frame += 1
        end
        
        Mirage.start_render_loop(render)
    catch e
        @error "Could not load 'cube.obj'. Make sure the file exists." e
    end
end

main()
```
