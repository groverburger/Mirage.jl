module Mirage

import GLFW
using ModernGL
using Revise
using LinearAlgebra # For matrix operations
using FileIO        # For loading images
using ImageIO       # For loading images (implicitly used by FileIO)
using Images
#using CairoMakie
# using Printf      # If needed for debugging text coords etc.

# === Existing OpenGL Helper Functions (Unchanged) ===

function gl_gen_one(gl_gen_fn)
    id = GLuint[0]
    gl_gen_fn(1, id)
    gl_check_error("generating a buffer, array, or texture")
    id[]
end

gl_gen_buffer() = gl_gen_one(glGenBuffers)
gl_gen_vertex_array() = gl_gen_one(glGenVertexArrays)
gl_gen_texture() = gl_gen_one(glGenTextures)

function get_info_log(obj::GLuint)
    is_shader = glIsShader(obj)
    get_iv = is_shader == GL_TRUE ? glGetShaderiv : glGetProgramiv
    get_info = is_shader == GL_TRUE ? glGetShaderInfoLog : glGetProgramInfoLog
    len = GLint[0]
    get_iv(obj, GL_INFO_LOG_LENGTH, len)
    max_length = len[]
    if max_length > 0
        buffer = zeros(GLchar, max_length)
        size_i = GLsizei[0]
        get_info(obj, max_length, size_i, buffer)
        len = size_i[]
        unsafe_string(pointer(buffer), len)
    else
        ""
    end
end

function validate_shader(shader)
    success = GLint[0]
    glGetShaderiv(shader, GL_COMPILE_STATUS, success)
    success[] == GL_TRUE
end

function gl_error_message()
    err = glGetError()
    err == GL_NO_ERROR ? "" :
        err == GL_INVALID_ENUM ? "GL_INVALID_ENUM" :
        err == GL_INVALID_VALUE ? "GL_INVALID_VALUE" :
        err == GL_INVALID_OPERATION ? "GL_INVALID_OPERATION" :
        err == GL_INVALID_FRAMEBUFFER_OPERATION ? "GL_INVALID_FRAMEBUFFER_OPERATION" :
        err == GL_OUT_OF_MEMORY ? "GL_OUT_OF_MEMORY" : "Unknown OpenGL error code $err."
end

function gl_check_error(action_name="")
    message = gl_error_message()
    if length(message) > 0
        error("OpenGL Error", isempty(action_name) ? "" : " during $action_name", ": ", message)
    end
end

function create_shader(source, typ)
    shader = glCreateShader(typ)::GLuint
    if shader == 0
        error("Error creating shader: ", gl_error_message())
    end
    glShaderSource(shader, 1, convert(Ptr{UInt8}, pointer([convert(Ptr{GLchar}, pointer(source))])), C_NULL)
    glCompileShader(shader)
    !validate_shader(shader) && error("Shader compilation error: ", get_info_log(shader))
    shader
end

function create_shader_program(f, vertex_shader, fragment_shader)
    prog = glCreateProgram()
    if prog == 0
        error("Error creating shader program: ", gl_error_message())
    end
    glAttachShader(prog, vertex_shader)
    gl_check_error("attaching vertex shader")
    glAttachShader(prog, fragment_shader)
    gl_check_error("attaching fragment shader")
    f(prog) # Optional: Bind attribute locations before linking
    glLinkProgram(prog)
    status = GLint[0]
    glGetProgramiv(prog, GL_LINK_STATUS, status)
    if status[] == GL_FALSE
        log = get_info_log(prog)
        glDeleteProgram(prog)
        error("Error linking shader program: ", log)
    end
    prog
end

create_shader_program(vertex_shader, fragment_shader) = create_shader_program((prog)->nothing, vertex_shader, fragment_shader)

# === New Global State & Constants ===

global glsl_version = ""
global window_width = 800
global window_height = 600
global projection_matrix = Matrix{Float32}(I, 4, 4)

# Shader sources
const simple_vertex_shader_source = """
    #version 330 core
    layout (location = 0) in vec2 aPos;

    uniform mat4 projection;

    void main()
    {
        gl_Position = projection * vec4(aPos.x, aPos.y, 0.0, 1.0);
    }
"""

const simple_fragment_shader_source = """
    #version 330 core
    out vec4 FragColor;

    uniform vec3 objectColor;

    void main()
    {
        FragColor = vec4(objectColor, 1.0);
    }
"""

const texture_vertex_shader_source = """
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoord;

    out vec2 TexCoord;

    uniform mat4 projection;

    void main()
    {
        gl_Position = projection * vec4(aPos.x, aPos.y, 0.0, 1.0);
        TexCoord = aTexCoord;
    }
"""

const texture_fragment_shader_source = """
    #version 330 core
    out vec4 FragColor;

    in vec2 TexCoord;

    uniform sampler2D textureSampler;
    uniform vec3 tintColor;

    void main()
    {
        vec4 texColor = texture(textureSampler, TexCoord);
        // Ensure transparent areas in texture remain transparent, tint opaque areas
        FragColor = vec4(texColor.rgb * tintColor, texColor.a);
        // For simple tinting without transparency: FragColor = texture(textureSampler, TexCoord) * vec4(tintColor, 1.0);
    }
"""

# Simple struct to hold shader programs and uniform locations
mutable struct ShaderInfo
    program_id::GLuint
    projection_loc::GLint
    color_loc::GLint      # For simple shader
    sampler_loc::GLint    # For texture shader
    tint_loc::GLint       # For texture shader
end

# === New Helper Functions ===

function create_context_info()
    global glsl_version
    # ... (rest of the function is unchanged from original)
    glsl = split(unsafe_string(glGetString(GL_SHADING_LANGUAGE_VERSION)), ['.', ' '])
    if length(glsl) >= 2
        glsl_num = VersionNumber(parse(Int, glsl[1]), parse(Int, glsl[2]))
        glsl_version = string(glsl_num.major) * rpad(string(glsl_num.minor),2,"0")
        # Enforce minimum version for shaders if needed, e.g., 330
        # if glsl_num < v"3.3"
        #     error("OpenGL version 3.3+ required (GLSL $glsl_version found)")
        # end
    else
        error("Unexpected version number string. Please report this bug! GLSL version string: $(glsl)")
    end

    glv_str = split(unsafe_string(glGetString(GL_VERSION)), ['.', ' '])
    if length(glv_str) >= 2
        glv = VersionNumber(parse(Int, glv_str[1]), parse(Int, glv_str[2]))
    else
        error("Unexpected version number string. Please report this bug! OpenGL version string: $(glv_str)")
    end
    dict = Dict{Symbol,Any}(
        :glsl_version   => glsl_num, # Use the parsed VersionNumber
        :gl_version     => glv,
        :gl_vendor	    => unsafe_string(glGetString(GL_VENDOR)),
        :gl_renderer	=> unsafe_string(glGetString(GL_RENDERER)),
    )
    return dict # Return the dict for info printing
end

# Orthographic projection matrix
# Maps x=[left, right] to [-1, 1] and y=[top, bottom] to [1, -1] (OpenGL coords)
function ortho(left::Float32, right::Float32, bottom::Float32, top::Float32, zNear::Float32 = -1.0f0, zFar::Float32 = 1.0f0)::Matrix{Float32}
    mat = zeros(Float32, 4, 4)
    mat[1, 1] = 2.0f0 / (right - left)
    mat[2, 2] = 2.0f0 / (top - bottom) # Flipped y-axis mapping
    mat[3, 3] = -2.0f0 / (zFar - zNear)
    mat[1, 4] = -(right + left) / (right - left)
    mat[2, 4] = -(top + bottom) / (top - bottom) # Flipped y-axis mapping
    mat[3, 4] = -(zFar + zNear) / (zFar - zNear)
    mat[4, 4] = 1.0f0
    return mat
end

function update_projection_matrix(width, height)
    global window_width, window_height, projection_matrix
    window_width = width
    window_height = height
    # Map pixel coords (0, width) -> (-1, 1) and (0, height) -> (1, -1)
    projection_matrix = ortho(0.0f0, Float32(width), Float32(height), 0.0f0)
    glViewport(0, 0, width, height)
end

function load_texture(filepath::String)::GLuint
    try
        img = FileIO.load(filepath)
        img_rgba = convert(Matrix{RGBA{N0f8}}, img) |> transpose
        return load_texture(img_rgba)
    catch e
        println("Error loading texture '$filepath': ", e)
        rethrow(e)
    end
end

function load_texture(img_rgba::Matrix{RGBA{N0f8}})::GLuint
    # Extract color components as Float32
    tex_width, tex_height = size(img_rgba)
    img_float32 = zeros(Float32, 4, tex_width, tex_height)  # Pre-allocate

    for y in 1:tex_height
        for x in 1:tex_width
            color = img_rgba[x, tex_height - y + 1]
            img_float32[1, x, y] = Float32(color.r)  # Red component
            img_float32[2, x, y] = Float32(color.g)  # Green component
            img_float32[3, x, y] = Float32(color.b)  # Blue component
            img_float32[4, x, y] = Float32(color.alpha) # Alpha component
        end
    end

    # Transpose and flip
    img_flipped = reverse(img_float32, dims=1) # Flip vertically

    tex_id = gl_gen_texture()
    glBindTexture(GL_TEXTURE_2D, tex_id)
    gl_check_error("binding texture")

    # Set texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    gl_check_error("setting wrap parameters")

    # Set texture filtering parameters
    #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR) # Use mipmaps for minification
    #glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR) # Linear for magnification
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST) # Use mipmaps for minification
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST) # Linear for magnification
    gl_check_error("setting filter parameters")

    # Upload the image data
    # Use GL_RGBA32F for Float32 RGBA data
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, tex_width, tex_height, 0, GL_RGBA, GL_FLOAT, img_flipped)
    gl_check_error("uploading texture data")

    # Generate mipmaps
    #glGenerateMipmap(GL_TEXTURE_2D)
    gl_check_error("generating mipmaps")

    glBindTexture(GL_TEXTURE_2D, 0) # Unbind
    return tex_id
end

# === Drawing Interface ===

mutable struct RenderContext
    simple_shader::ShaderInfo
    texture_shader::ShaderInfo
    vao::GLuint
    vbo::GLuint
    font_texture::GLuint
    char_width::Float32  # Assuming fixed width font atlas grid cell
    char_height::Float32 # Assuming fixed height font atlas grid cell
    atlas_cols::Int      # Number of columns in font atlas grid
    atlas_rows::Int      # Number of rows in font atlas grid
end

function init_render_context()::RenderContext
    # Compile Shaders
    simple_vs = create_shader(simple_vertex_shader_source, GL_VERTEX_SHADER)
    simple_fs = create_shader(simple_fragment_shader_source, GL_FRAGMENT_SHADER)
    simple_program = create_shader_program(simple_vs, simple_fs)
    glDeleteShader(simple_vs)
    glDeleteShader(simple_fs)

    texture_vs = create_shader(texture_vertex_shader_source, GL_VERTEX_SHADER)
    texture_fs = create_shader(texture_fragment_shader_source, GL_FRAGMENT_SHADER)
    texture_program = create_shader_program(texture_vs, texture_fs)
    glDeleteShader(texture_vs)
    glDeleteShader(texture_fs)

    simple_shader_info = ShaderInfo(
        simple_program,
        glGetUniformLocation(simple_program, "projection"),
        glGetUniformLocation(simple_program, "objectColor"),
        -1, -1 # Not used
    )
    texture_shader_info = ShaderInfo(
        texture_program,
        glGetUniformLocation(texture_program, "projection"),
        -1, # Not used
        glGetUniformLocation(texture_program, "textureSampler"),
        glGetUniformLocation(texture_program, "tintColor")
    )

    # Create VAO and VBO for dynamic drawing (shared)
    vao = gl_gen_vertex_array()
    vbo = gl_gen_buffer()
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    # Position attribute (vec2)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), C_NULL)
    glEnableVertexAttribArray(0)
    # Texture coord attribute (vec2) - stride is 4 floats (pos.x, pos.y, tex.u, tex.v)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), Ptr{Cvoid}(2 * sizeof(GLfloat)))
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    # --- Font Setup (Simplified Placeholder) ---
    # In a real app, load a font atlas texture and metrics file (e.g., using FreeType)
    # Here, we'll just create a dummy 1x1 white texture and define some basic grid params
    dummy_font_tex = gl_gen_texture()
    glBindTexture(GL_TEXTURE_2D, dummy_font_tex)
    white_pixel = Float32[1.0, 1.0, 1.0, 1.0]
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1, 1, 0, GL_RGBA, GL_FLOAT, white_pixel)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glBindTexture(GL_TEXTURE_2D, 0)
    # Pretend we have a 16x8 grid font atlas (like ASCII)
    char_w, char_h = 8.0f0, 14.0f0 # Example pixel dimensions of a character *cell*
    atlas_cols, atlas_rows = 16, 6

    return RenderContext(
        simple_shader_info,
        texture_shader_info,
        vao, vbo,
        dummy_font_tex, # Use dummy texture ID
        char_w, char_h,
        atlas_cols, atlas_rows
    )
end

function cleanup_render_context(ctx::RenderContext)
    glDeleteProgram(ctx.simple_shader.program_id)
    glDeleteProgram(ctx.texture_shader.program_id)
    glDeleteBuffers(1, [ctx.vbo])
    glDeleteVertexArrays(1, [ctx.vao])
    glDeleteTextures(1, [ctx.font_texture]) # Delete font texture too
end


# Draw a solid color rectangle
function draw_rectangle(ctx::RenderContext, x::Float32, y::Float32, w::Float32, h::Float32, color::Vector{Float32})
    glUseProgram(ctx.simple_shader.program_id)
    glUniformMatrix4fv(ctx.simple_shader.projection_loc, 1, GL_FALSE, projection_matrix)
    glUniform3f(ctx.simple_shader.color_loc, color[1], color[2], color[3])

    # Define vertices (x, y, u, v) - u,v are ignored by this shader but needed for shared VBO structure
    # Top-left, bottom-left, bottom-right, top-right (for two triangles)
    vertices = GLfloat[
        x, y,      0.0, 0.0,  # Top-left
        x, y + h,  0.0, 1.0,  # Bottom-left
        x + w, y + h,  1.0, 1.0,  # Bottom-right

        x, y,      0.0, 0.0,  # Top-left
        x + w, y + h,  1.0, 1.0,  # Bottom-right
        x + w, y,      1.0, 0.0   # Top-right
    ]

    glBindVertexArray(ctx.vao)
    glBindBuffer(GL_ARRAY_BUFFER, ctx.vbo)
    # Upload data for this specific rectangle draw call
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW)

    glDrawArrays(GL_TRIANGLES, 0, 6) # 6 vertices for 2 triangles

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    glUseProgram(0)
end

# Draw a textured rectangle
function draw_textured_rectangle(ctx::RenderContext, x::Float32, y::Float32, w::Float32, h::Float32, texture_id::GLuint, tint_color::Vector{Float32}=[1.0f0, 1.0f0, 1.0f0])
    glUseProgram(ctx.texture_shader.program_id)
    glUniformMatrix4fv(ctx.texture_shader.projection_loc, 1, GL_FALSE, projection_matrix)
    glUniform3f(ctx.texture_shader.tint_loc, tint_color[1], tint_color[2], tint_color[3])

    glActiveTexture(GL_TEXTURE0) # Activate texture unit 0
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glUniform1i(ctx.texture_shader.sampler_loc, 0) # Tell sampler to use texture unit 0

    # Define vertices (x, y, u, v)
    # Note: Standard texture coords map (0,0) bottom-left, (1,1) top-right
    # If your image loading flips, adjust UV coords accordingly. Assumes standard coords here.
    vertices = GLfloat[
        x, y,      0.0, 1.0,  # Top-left
        x, y + h,  0.0, 0.0,  # Bottom-left
        x + w, y + h,  1.0, 0.0,  # Bottom-right

        x, y,      0.0, 1.0,  # Top-left
        x + w, y + h,  1.0, 0.0,  # Bottom-right
        x + w, y,      1.0, 1.0   # Top-right
    ]

    glBindVertexArray(ctx.vao)
    glBindBuffer(GL_ARRAY_BUFFER, ctx.vbo)
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW)

    glDrawArrays(GL_TRIANGLES, 0, 6)

    glBindTexture(GL_TEXTURE_2D, 0) # Unbind texture
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    glUseProgram(0)
end

# Draw text using the loaded font atlas (simplified)
function draw_text(ctx::RenderContext, text::String, x_start::Float32, y_start::Float32, scale::Float32, color::Vector{Float32})
    glUseProgram(ctx.texture_shader.program_id)
    glUniformMatrix4fv(ctx.texture_shader.projection_loc, 1, GL_FALSE, projection_matrix)
    glUniform3f(ctx.texture_shader.tint_loc, color[1], color[2], color[3])

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, ctx.font_texture) # Use the font texture
    glUniform1i(ctx.texture_shader.sampler_loc, 0)

    glBindVertexArray(ctx.vao)
    glBindBuffer(GL_ARRAY_BUFFER, ctx.vbo)

    vertices = GLfloat[]
    x_cursor = x_start

    # Simplified: Assume ASCII, fixed grid, no bearing/kerning
    atlas_cell_w_uv = 1.0f0 / ctx.atlas_cols
    atlas_cell_h_uv = 1.0f0 / ctx.atlas_rows

    for char in text
        if isascii(char)
            char_code = Int(char) - 32
            # Calculate grid position (row, col)
            col = char_code % ctx.atlas_cols
            row = char_code ÷ ctx.atlas_cols

            # Calculate UV coordinates for this character cell
            # UV origin (0,0) is bottom-left in OpenGL textures
            u0 = Float32(col) * atlas_cell_w_uv
            v0 = 1.0f0 - Float32(row + 1) * atlas_cell_h_uv # Y is flipped
            u1 = u0 + atlas_cell_w_uv
            v1 = v0 + atlas_cell_h_uv

            # Calculate screen position and size for this character's quad
            char_render_w = ctx.char_width * scale
            char_render_h = ctx.char_height * scale
            xpos = x_cursor
            ypos = y_start # Simple baseline alignment

            # Define quad vertices (x, y, u, v) - 6 vertices for 2 triangles
            append!(vertices, GLfloat[
                xpos, ypos,                   u0, v1,  # Top-left
                xpos, ypos + char_render_h,   u0, v0,  # Bottom-left
                xpos + char_render_w, ypos + char_render_h, u1, v0,  # Bottom-right

                xpos, ypos,                   u0, v1,  # Top-left
                xpos + char_render_w, ypos + char_render_h, u1, v0,  # Bottom-right
                xpos + char_render_w, ypos,   u1, v1   # Top-right
            ])

            # Advance cursor (simplified fixed width advance)
            x_cursor += char_render_w
        else
            # Skip non-ASCII or handle differently
            x_cursor += (ctx.char_width * scale) # Advance by space width
        end
    end

    if !isempty(vertices)
        # Upload all vertex data for the entire string at once
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW)
        # Draw all characters
        glDrawArrays(GL_TRIANGLES, 0, length(vertices) ÷ 4) # 4 floats per vertex
    end

    glBindTexture(GL_TEXTURE_2D, 0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    glUseProgram(0)
end

# === Main Application Logic ===

function julia_main()::Cint
    # --- Initialization ---
    if !GLFW.Init()
        error("GLFW initialization failed")
        return -1
    end

    # Request OpenGL 3.3 Core context
    GLFW.WindowHint(GLFW.CONTEXT_VERSION_MAJOR, 3)
    GLFW.WindowHint(GLFW.CONTEXT_VERSION_MINOR, 3)
    GLFW.WindowHint(GLFW.OPENGL_PROFILE, GLFW.OPENGL_CORE_PROFILE)
    GLFW.WindowHint(GLFW.OPENGL_FORWARD_COMPAT, GL_TRUE) # Required on macOS

    # Create a windowed mode window and its OpenGL context
    window = GLFW.CreateWindow(window_width, window_height, "Julia OpenGL Shapes & Text Demo")
    if window == C_NULL
        GLFW.Terminate()
        error("Failed to create GLFW window")
        return -1
    end
    GLFW.MakeContextCurrent(window)
    GLFW.ShowWindow(window)

    # Setup callbacks
    GLFW.SetFramebufferSizeCallback(window, (_, w, h) -> update_projection_matrix(w, h))

    # Enable VSync
    GLFW.SwapInterval(1)

    # Initial projection matrix setup
    update_projection_matrix(window_width, window_height)

    # Query and print OpenGL info
    @info "OpenGL Context Info:" create_context_info()

    # Initialize render context (shaders, VAO/VBO)
    render_ctx = init_render_context()
    render_ctx.font_texture = load_texture("./ascii_font_atlas.png")

    # --- Load Assets ---
    # Example: Load a texture (provide a path to an actual image file)
    # Create a dummy texture if no file exists
    test_texture_id = GLuint(0)
    try
        # IMPORTANT: Replace with the actual path to your image file
        test_texture_path = "test_texture.png" # Example path
        if isfile(test_texture_path)
            test_texture_id = load_texture(test_texture_path)
            @info "Loaded test texture: $test_texture_path (ID: $test_texture_id)"
        else
            @warn "Test texture file not found: $test_texture_path. Skipping texture demo."
             # Create a simple 2x2 checkerboard texture programmatically as fallback
            checker_data = Float32[
                0.0, 0.0, 0.0, 1.0,  1.0, 1.0, 1.0, 1.0, # Black, White
                1.0, 1.0, 1.0, 1.0,  0.0, 0.0, 0.0, 1.0  # White, Black
            ]
            test_texture_id = gl_gen_texture()
            glBindTexture(GL_TEXTURE_2D, test_texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 2, 2, 0, GL_RGBA, GL_FLOAT, checker_data)
            glBindTexture(GL_TEXTURE_2D, 0)
            @info "Created fallback checkerboard texture (ID: $test_texture_id)"
        end
    catch e
        @error "Failed to load or create texture: $e"
        # Continue without texture
    end

    # --- Create and Render Makie Plot to Texture ---
    makie_texture_id = GLuint(0)
    makie_plot_width_px = 300
    makie_plot_height_px = 200
    try
        CairoMakie.activate!(type = "png") # Specify backend type for rendering buffer
        # Create a figure with specified pixel size
        makie_fig = CairoMakie.Figure(size = (makie_plot_width_px, makie_plot_height_px))
        ax = CairoMakie.Axis(makie_fig[1, 1], title="Simple Plot")
        CairoMakie.scatter!(ax, 1:10, rand(10) .* 5, color=:orange, markersize=8)
        CairoMakie.lines!(ax, 1:10, rand(10) .* 5, color=:cyan)
        # Render the figure to an in-memory buffer
        # This needs Makie v0.17+; older versions might need save/load workaround
        makie_img_buffer = CairoMakie.colorbuffer(makie_fig) |> transpose # Returns Matrix{RGBA{N0f8}}

        # Load this buffer into an OpenGL texture
        makie_texture_id = load_texture(RGBA{N0f8}.(makie_img_buffer))
        @info "Generated Makie plot texture (ID: $makie_texture_id)"
    catch e
        @error "Failed to generate Makie plot texture: $e"
        showerror(stdout, e, catch_backtrace())
        println()
        # Continue without the Makie plot texture
    end
    # --- End Makie Plot ---

    # --- Main Loop ---
    frame_count::Int64 = 0
    last_frame_time = time() # Initialize time measurement before the loop

    # Enable blending for text/texture transparency
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    global terminate = function ()
        @info "Cleaning up resources..."
        cleanup_render_context(render_ctx)
        if test_texture_id != 0
            glDeleteTextures(1, [test_texture_id])
        end
        GLFW.DestroyWindow(window)
        GLFW.Terminate()
        @info "Shutdown complete."
    end

    while !GLFW.WindowShouldClose(window)
        try
            frame_count += 1
            current_frame_time = time() # Get time at the start of the frame processing
            delta_time = current_frame_time - last_frame_time
            last_frame_time = current_frame_time

            # --- Rendering ---
            bg_color = 0.1f0 # Gray background
            glClearColor(bg_color, bg_color, bg_color, 1.0f0)
            glClear(GL_COLOR_BUFFER_BIT)

            # Demo drawing calls:
            # Draw a solid red rectangle
            draw_rectangle(render_ctx, 50.0f0, 50.0f0, 100.0f0, 80.0f0, [1.0f0, 0.0f0, 0.0f0])

            # Draw a solid green rectangle
            draw_rectangle(render_ctx, 200.0f0, 100.0f0, 50.0f0, 150.0f0, [0.0f0, 1.0f0, 0.0f0])

            # Draw the loaded texture (if available)
            if test_texture_id != 0
                # Full size
                draw_textured_rectangle(render_ctx, 50.0f0, 200.0f0, 150.0f0, 150.0f0, render_ctx.font_texture)
                # Tinted blue and scaled
                draw_textured_rectangle(render_ctx, 250.0f0, 250.0f0, 100.0f0, 100.0f0, test_texture_id, [0.5f0, 0.5f0, 1.0f0])
            end

            # Calculate FPS using delta_time (add epsilon to prevent division by zero)
            fps = round(Int, 1.0 / (delta_time + 1e-9))

            # Draw text (using the simplified font renderer)
            draw_text(render_ctx, "Hello Julia OpenGL!", 50.0f0, 400.0f0, 2.0f0, [1.0f0, 1.0f0, 0.0f0]) # Yellow, scaled up
            draw_text(render_ctx, "FPS: $fps", 10.0f0, 10.0f0, 1.0f0, [1.0f0, 1.0f0, 1.0f0]) # White, normal size
            draw_text(render_ctx, "0123456789 ASCII /?!", 50.0f0, 450.0f0, 2.0f0, [0.0f0, 1.0f0, 1.0f0]) # Cyan

            # --- Draw the Makie plot texture ---
            #=
            if makie_texture_id != 0
                # Position it, e.g., top-right corner with a margin
                plot_margin = 10.0f0
                plot_x_pos = Float32(window_width) - Float32(makie_plot_width_px) * 2 - plot_margin
                plot_y_pos = plot_margin
                draw_textured_rectangle(render_ctx, plot_x_pos, plot_y_pos, Float32(makie_plot_width_px) * 2, Float32(makie_plot_height_px) * 2, makie_texture_id)
            end
            =#

            # --- End Frame ---
            GLFW.SwapBuffers(window)
            GLFW.PollEvents()

            gl_check_error("end of frame $frame_count") # Check for errors each frame
        catch e
            println("Error in main loop: ", e)
            showerror(stdout, e, catch_backtrace())
            println()
            sleep(0.5)
            #return 1
        end
    end

    terminate()

    return 0
end

# Exported function already snake_case
export julia_main

end # module GraphicsTest

