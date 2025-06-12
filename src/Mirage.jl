module Mirage

import GLFW
using ModernGL
using Revise
using FileIO
using ImageIO
using Images

function translate!(matrix::Matrix{T}, tx::Real, ty::Real, tz::Real = 0.0) where T
    translation = T[
        1.0 0.0 0.0 tx;
        0.0 1.0 0.0 ty;
        0.0 0.0 1.0 tz;
        0.0 0.0 0.0 1.0
    ]
    result = matrix * translation
    for i in 1:size(matrix, 1), j in 1:size(matrix, 2)
        matrix[i, j] = result[i, j]
    end
    return matrix
end

function rotate!(matrix::Matrix{T}, angle::Real) where T
    c = cos(angle)
    s = sin(angle)
    rotation = T[
        c   -s    0.0  0.0;
        s    c    0.0  0.0;
        0.0  0.0  1.0  0.0;
        0.0  0.0  0.0  1.0
    ]
    result = matrix * rotation
    for i in 1:size(matrix, 1), j in 1:size(matrix, 2)
        matrix[i, j] = result[i, j]
    end
    return matrix
end

function rotate!(matrix::Matrix{T}, angle::Real, axis::Vector{T}) where T
    # Validate axis input
    length(axis) == 3 || throw(ArgumentError("Axis must be 3-element vector"))
    
    # Manual normalization
    norm = sqrt(sum(x -> x^2, axis))
    norm ≈ 0 && throw(ArgumentError("Rotation axis cannot be zero vector"))
    axis_normalized = axis ./ norm

    # Rotation matrix components
    c = cos(angle)
    s = sin(angle)
    t = 1 - c
    x, y, z = axis_normalized

    # Construct rotation matrix
    rotation = T[
        t*x^2 + c      t*x*y - s*z   t*x*z + s*y   0.0;
        t*x*y + s*z    t*y^2 + c     t*y*z - s*x   0.0;
        t*x*z - s*y    t*y*z + s*x   t*z^2 + c     0.0;
        0.0            0.0           0.0           1.0
    ]

    # In-place matrix update
    result = matrix * rotation
    for i in 1:size(matrix, 1), j in 1:size(matrix, 2)
        matrix[i, j] = result[i, j]
    end
    return matrix
end

function scale!(matrix::Matrix{T}, sx::Real, sy::Real, sz::Real = 1.0) where T
    scaling = T[
        sx  0.0  0.0  0.0;
        0.0  sy   0.0  0.0;
        0.0  0.0  sz   0.0;
        0.0  0.0  0.0  1.0
    ]
    result = matrix * scaling
    for i in 1:size(matrix, 1), j in 1:size(matrix, 2)
        matrix[i, j] = result[i, j]
    end
    return matrix
end

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
    shader::GLuint = glCreateShader(typ)
    if shader == 0
        error("Error creating shader: ", gl_error_message())
    end
    glShaderSource(shader, 1, convert(Ptr{UInt8}, pointer([convert(Ptr{GLchar}, pointer(source))])), C_NULL)
    glCompileShader(shader)
    !validate_shader(shader) && error("Shader compilation error: ", get_info_log(shader))
    return shader
end

function create_shader_program(vertex_shader, fragment_shader)
    prog::GLuint = glCreateProgram()
    if prog == 0
        error("Error creating shader program: ", gl_error_message())
    end
    glAttachShader(prog, vertex_shader)
    gl_check_error("attaching vertex shader")
    glAttachShader(prog, fragment_shader)
    gl_check_error("attaching fragment shader")
    glLinkProgram(prog)
    status = GLint[0]
    glGetProgramiv(prog, GL_LINK_STATUS, status)
    if status[] == GL_FALSE
        log = get_info_log(prog)
        glDeleteProgram(prog)
        error("Error linking shader program: ", log)
    end
    return prog
end

global glsl_version = ""
global immediate_mesh = nothing

const texture_vertex_shader_source = """
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec2 aTexCoord;

    out vec2 TexCoord;

    uniform mat4 projection;
    uniform mat4 view;
    uniform mat4 model;

    void main()
    {
        gl_Position = projection * view * model * vec4(aPos.x, aPos.y, aPos.z, 1.0);
        TexCoord = aTexCoord;
    }
"""

const texture_fragment_shader_source = """
    #version 330 core
    out vec4 FragColor;

    in vec2 TexCoord;

    uniform sampler2D textureSampler;
    uniform vec4 color;

    void main()
    {
        vec4 texColor = texture(textureSampler, TexCoord) * color;
        if (texColor.a == 0.0) { discard; }
        FragColor = texColor;
    }
"""

# Simple struct to hold shader programs and uniform locations
struct ShaderInfo
    program_id::GLuint
    uniform_locations::Dict{String, GLint}
end

function initialize_shader_uniform!(shader::ShaderInfo, uniform_name::String)
    shader.uniform_locations[uniform_name] = glGetUniformLocation(shader.program_id, uniform_name)
end

# === New Helper Functions ===

function create_context_info()
    global glsl_version
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

function perspective(fov::Float32, aspectRatio::Float32, near::Float32, far::Float32)::Matrix{Float32}
    top = near * tan(fov/2)
    bottom = -1*top
    right = top * aspectRatio
    left = -1*right

    return Float32[
        2*near/(right-left) 0                   (right+left)/(right-left) 0;
        0                   2*near/(top-bottom) (top+bottom)/(top-bottom) 0;
        0                   0                   -1*(far+near)/(far-near) -2*far*near/(far-near);
        0                   0                   -1                        0
    ]
end

function view(position, target, up = [0, 0, 1])
  z = normalize(position - target)
  x = normalize(cross(up, z))
  y = cross(z, x)

  return Float32[
      x[1] x[2] x[3] -dot(x, position);
      y[1] y[2] y[3] -dot(y, position);
      z[1] z[2] z[3] -dot(z, position);
      0    0    0    1
  ]
end

function normalize(v::Vector{Float32})::Vector{Float32}
    len = sqrt(sum(v .^ 2))
    return len > 0.0f0 ? v ./ len : v
end

function cross(a::Vector{Float32}, b::Vector{Float32})::Vector{Float32}
    return Float32[
        a[2] * b[3] - a[3] * b[2],
        a[3] * b[1] - a[1] * b[3],
        a[1] * b[2] - a[2] * b[1]
    ]
end

function dot(a::Vector{Float32}, b::Vector{Float32})::Float32
    return sum(a .* b)
end

function update_ortho_projection_matrix(width=get_context().width,
                                        height=get_context().height,
                                        dpi_scaling=get_context().dpi_scaling)
    # Map pixel coords (0, width) -> (-1, 1) and (0, height) -> (1, -1)
    get_state().projection = ortho(0.0f0, Float32(width / dpi_scaling), Float32(height / dpi_scaling), 0.0f0)
    glViewport(0, 0, width, height)
end

function update_perspective_projection_matrix(width=get_context().width,
                                              height=get_context().height,
                                              dpi_scaling=get_context().dpi_scaling)
    get_state().projection = perspective(Float32(pi / 4), Float32(width / height), 0.001f0, 10_000f0)
    glViewport(0, 0, width, height)
end

function load_texture(filepath::String)::GLuint
    try
        img = FileIO.load(filepath)
        img_rgba = convert(Matrix{RGBA{N0f8}}, img)
        return load_texture(img_rgba)
    catch e
        println("Error loading texture '$filepath': ", e)
        rethrow(e)
    end
end

function load_texture(img_rgba::Matrix{Images.RGBA{Images.N0f8}})::GLuint
    tex_height, tex_width = size(img_rgba)

    # Combine flip and transpose in a single operation
    output_img = permutedims(img_rgba[end:-1:1, :], (2, 1))

    # Convert to bytes
    output_bytes = vec(reinterpret(UInt8, output_img))

    # OpenGL texture creation
    tex_id = Mirage.gl_gen_texture()
    glBindTexture(GL_TEXTURE_2D, tex_id)
    gl_check_error("binding texture")

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    gl_check_error("setting wrap parameters")

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    gl_check_error("setting filter parameters")

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, tex_width, tex_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, output_bytes)
    glBindTexture(GL_TEXTURE_2D, 0)
    gl_check_error("uploading texture data")

    return tex_id
end

# === Drawing Interface ===

@kwdef mutable struct ContextState
    transform::Matrix{Float32} = Float32[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
    view::Matrix{Float32} = Float32[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
    projection::Matrix{Float32} = ortho(0f0, 800f0, 600f0, 0f0)
    fill_color::Tuple{Float32, Float32, Float32, Float32} = (1, 1, 1, 1)
    stroke_color::Tuple{Float32, Float32, Float32, Float32} = (0, 0, 0, 1)
    stroke_width::Float32 = 1
    stroke_path::Vector{Tuple{Float32, Float32}} = Tuple{Float32, Float32}[]
    stroke_cursor::Tuple{Float32, Float32} = (0, 0)
end

function clone(x::ContextState)
    fields = fieldnames(ContextState)
    kwargs = Dict{Symbol, Any}(field => deepcopy(getfield(x, field)) for field in fields)
    return ContextState(; kwargs...)
end

mutable struct RenderContext
    shader::ShaderInfo
    blank_texture::GLuint
    font_texture::GLuint
    char_width::Float32  # Assuming fixed width font atlas grid cell
    char_height::Float32 # Assuming fixed height font atlas grid cell
    atlas_cols::Int      # Number of columns in font atlas grid
    atlas_rows::Int      # Number of rows in font atlas grid
    context_stack::Vector{ContextState}
    width::Int
    height::Int
    dpi_scaling::Number

    function RenderContext()::RenderContext
        # Compile Shaders
        texture_vs = create_shader(texture_vertex_shader_source, GL_VERTEX_SHADER)
        texture_fs = create_shader(texture_fragment_shader_source, GL_FRAGMENT_SHADER)
        texture_program = create_shader_program(texture_vs, texture_fs)
        glDeleteShader(texture_vs)
        glDeleteShader(texture_fs)

        shader = ShaderInfo(texture_program, Dict{String, GLint}())
        initialize_shader_uniform!(shader, "projection")
        initialize_shader_uniform!(shader, "view")
        initialize_shader_uniform!(shader, "model")
        initialize_shader_uniform!(shader, "textureSampler")
        initialize_shader_uniform!(shader, "color")

        blank_texture = gl_gen_texture()
        font_texture = load_texture("./ascii_font_atlas.png")

        # --- Font Setup  ---
        glBindTexture(GL_TEXTURE_2D, blank_texture)
        white_pixel = Float32[1.0, 1.0, 1.0, 1.0]
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1, 1, 0, GL_RGBA, GL_FLOAT, white_pixel)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glBindTexture(GL_TEXTURE_2D, 0)
        char_width, char_height = 8.0f0, 14.0f0 # Pixel dimensions of a character cell
        atlas_cols, atlas_rows = 16, 6

        return new(
            shader,
            blank_texture,
            font_texture,
            char_width, char_height,
            atlas_cols, atlas_rows,
            [ContextState()]
        )
    end
end

const render_context = Ref{RenderContext}()

function cleanup_render_context(ctx::RenderContext = get_context())
    glDeleteProgram(ctx.shader.program_id)
    glDeleteTextures(1, [ctx.blank_texture])
    glDeleteTextures(1, [ctx.font_texture])
    global immediate_mesh = nothing
end

save() = push!(get_context().context_stack, clone(get_context().context_stack[end]))

restore() = pop!(get_context().context_stack)

get_context() = render_context[]

get_state() = get_context().context_stack[end]

function translate(dx::Number, dy::Number, dz::Number = 0)
    translate!(get_state().transform, dx, dy, dz)
end

function scale(dx::Number, dy::Number, dz::Number = 1)
    scale!(get_state().transform, dx, dy, dz)
end
scale(n::Number) = scale(n, n, n)

function rotate(angle::Number)
    rotate!(get_state().transform, angle)
end

function rotate(x::Number, y::Number, z::Number)
    rotate!(get_state().transform, x, Float32[1, 0, 0])
    rotate!(get_state().transform, y, Float32[0, 1, 0])
    rotate!(get_state().transform, z, Float32[0, 0, 1])
end

function lookat(args...)
    get_state().view = view(args...)
end

function beginpath()
    empty!(get_state().stroke_path)
end

function moveto(x::Number, y::Number)
    get_state().stroke_cursor = (Float32(x), Float32(y))
end

function lineto(x::Number, y::Number)
    last_cursor::Tuple{Float32, Float32} = get_state().stroke_cursor
    now_cursor::Tuple{Float32, Float32} = (Float32(x), Float32(y))
    get_state().stroke_cursor = now_cursor
    push!(get_state().stroke_path, last_cursor)
    push!(get_state().stroke_path, now_cursor)
end

function fillcolor(tuple::Tuple{Number, Number, Number})
    get_state().fill_color = (
        Float32(tuple[1]),
        Float32(tuple[2]),
        Float32(tuple[3]),
        Float32(1)
    )
end

function fillcolor(tuple::Tuple{Number, Number, Number, Number})
    get_state().fill_color = (
        Float32(tuple[1]),
        Float32(tuple[2]),
        Float32(tuple[3]),
        Float32(tuple[4])
    )
end

function strokecolor(tuple::Tuple{Number, Number, Number})
    get_state().stroke_color = (
        Float32(tuple[1]),
        Float32(tuple[2]),
        Float32(tuple[3]),
        Float32(1)
    )
end

function strokecolor(tuple::Tuple{Number, Number, Number, Number})
    get_state().stroke_color = (
        Float32(tuple[1]),
        Float32(tuple[2]),
        Float32(tuple[3]),
        Float32(tuple[4])
    )
end

function rgba(r::Int, g::Int, b::Int, a::Int = 255)::Tuple{Float32, Float32, Float32, Float32}
    return (r / 255, g / 255, b / 255, a / 255)
end

function strokewidth(w::Number)
    get_state().stroke_width = w
end

function stroke()
    immediate_mesh = get_immediate_mesh()
    state::ContextState = get_state()
    if length(state.stroke_path) < 2
        @debug "Skipping stroke with insufficient path points"
        return
    end
    vertices::Vector{Float32} = Float32[]

    for i in 1:2:length(state.stroke_path)
        # Draw rectangle for line
        (x1, y1) = state.stroke_path[i]
        (x2, y2) = state.stroke_path[i + 1]
        angle::Float64 = atan(y2 - y1, x2 - x1)
        dx::Float32 = cos(angle + pi / 2) * state.stroke_width
        dy::Float32 = sin(angle + pi / 2) * state.stroke_width
        append!(vertices, Float32[
            x1 + dx, y1 + dy, 0.0, 1.0,
            x1 - dx, y1 - dy, 0.0, 0.0,
            x2 + dx, y2 + dy, 1.0, 0.0,
            x1 - dx, y1 - dy, 0.0, 1.0,
            x2 + dx, y2 + dy, 1.0, 0.0,
            x2 - dx, y2 - dy, 1.0, 1.0
        ])

        # Add simple elbows to join connecting lines
        if i + 1 >= length(state.stroke_path);
            continue
        end
        (x3, y3) = state.stroke_path[i + 2]
        if isapprox(x2, x3) && isapprox(y2, y3)
            (x4, y4) = state.stroke_path[i + 3]
            next_angle::Float64 = atan(y4 - y3, x4 - x3)
            ndx::Float32 = cos(next_angle + pi / 2) * state.stroke_width
            ndy::Float32 = sin(next_angle + pi / 2) * state.stroke_width
            append!(vertices, Float32[
                x2 + dx,  y2 + dy, 0.0, 1.0,
                x2 - dx,  y2 - dy, 0.0, 0.0,
                x2 + ndx, y2 + ndy, 1.0, 0.0,
                x2 - dx,  y2 - dy, 0.0, 1.0,
                x2 + ndx, y2 + ndy, 1.0, 0.0,
                x2 - ndx, y2 - ndy, 1.0, 1.0
            ])
        end
    end

    update_mesh_vertices!(immediate_mesh, vertices)
    draw_mesh(immediate_mesh, get_context().blank_texture, [state.stroke_color...])
end

function fill()
    immediate_mesh = get_immediate_mesh()
    state::ContextState = get_state()

    unique_points = unique(state.stroke_path)
    x3::Number = 0
    y3::Number = 0
    for (x, y) in unique_points
        x3 += x
        y3 += y
    end
    x3 /= length(unique_points)
    y3 /= length(unique_points)

    for i in 1:2:length(state.stroke_path)
        # Draw rectangle for line
        (x1, y1) = state.stroke_path[i]
        (x2, y2) = state.stroke_path[i + 1]
        update_mesh_vertices!(immediate_mesh, Float32[
            x1, y1, 0.0, 1.0,
            x2, y2, 0.0, 0.0,
            x3, y3, 1.0, 0.0
        ])
        draw_mesh(immediate_mesh, get_context().blank_texture, [state.fill_color...])
    end
end

function rect(x::Number, y::Number, w::Number, h::Number)
    moveto(x, y)
    lineto(x + w, y)
    lineto(x + w, y + h)
    lineto(x, y + h)
    lineto(x, y)
end

function circle(r::Number, x::Number = 0, y::Number = 0, segments::Int = 32)
    for i in 1:segments
        angle::Float32 = 2.0f0 * π * (i - 1) / segments
        next_angle::Float32 = 2.0f0 * π * i / segments
        x::Float32 = r * cos(angle)
        y::Float32 = r * sin(angle)
        if i == 1
            moveto(x, y)
        else
            lineto(x, y)
        end
    end
    lineto(r * cos(1 / segments), r * sin(1 / segments))
end

function fillrect(x::Number, y::Number, w::Number, h::Number)
    drawimage(x, y, w, h, get_context().blank_texture)
end

function fillcircle(radius::Number, x::Number = 0, y::Number = 0, segments::Int = 32)
    # Create triangles by connecting center to each pair of consecutive vertices
    # This creates a fan-like triangulation
    vertices = Float32[]

    for i in 1:segments
        angle::Float32 = 2.0f0 * π * (i - 1) / segments
        next_angle::Float32 = 2.0f0 * π * i / segments

        # Center point
        append!(vertices, 0.0f0, 0.0f0)
        append!(vertices, 0.5f0, 0.5f0)

        # Current outer point
        append!(vertices, radius * cos(angle), radius * sin(angle))
        append!(vertices, cos(angle) * 0.5f0 + 0.5f0, sin(angle) * 0.5f0 + 0.5f0)

        # Next outer point
        append!(vertices, radius * cos(next_angle), radius * sin(next_angle))
        append!(vertices, cos(next_angle) * 0.5f0 + 0.5f0, sin(next_angle) * 0.5f0 + 0.5f0)
    end

    immediate_mesh = get_immediate_mesh()
    state::ContextState = get_state()
    update_mesh_vertices!(immediate_mesh, vertices)
    draw_mesh(immediate_mesh, get_context().blank_texture, [state.fill_color...])
end

function drawimage(x::Number,
                   y::Number,
                   w::Number,
                   h::Number,
                   texture_id::GLuint)
    immediate_mesh = get_immediate_mesh()
    update_mesh_vertices!(immediate_mesh, Float32[
        x, y,          0.0, 1.0,  # Top-left
        x, y + h,      0.0, 0.0,  # Bottom-left
        x + w, y + h,  1.0, 0.0,  # Bottom-right
        x, y,          0.0, 1.0,  # Top-left
        x + w, y + h,  1.0, 0.0,  # Bottom-right
        x + w, y,      1.0, 1.0   # Top-right
    ])
    draw_mesh(immediate_mesh, texture_id)
end

# Draw text using the loaded font atlas (simplified)
function text(text::String)
    ctx::RenderContext = get_context()
    vertices = GLfloat[]
    x_cursor = 0f0

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
            char_render_w = ctx.char_width
            char_render_h = ctx.char_height
            xpos = x_cursor
            ypos = 0f0 # Simple baseline alignment

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
            x_cursor += ctx.char_width # Advance by space width
        end
    end

    if !isempty(vertices)
        # Upload all vertex data for the entire string at once
        immediate_mesh = get_immediate_mesh()
        update_mesh_vertices!(immediate_mesh, vertices)
        # Draw all characters
        draw_mesh(immediate_mesh, ctx.font_texture, [get_state().fill_color...])
    end
end

function get_immediate_mesh()
    global immediate_mesh
    if immediate_mesh == nothing || immediate_mesh.vao == 0
        @debug "Creating new immediate mesh"
        immediate_mesh = create_mesh()
    end
    return immediate_mesh
end

include("./meshes.jl")

# === Main Application Logic ===

const window = Ref{GLFW.Window}()

function set_render_context(ctx::RenderContext)
    render_context[] = ctx
end

function initialize_render_context()
    set_render_context(RenderContext())
end

function initialize(;window_width::Int = 800, window_height::Int = 600)
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
    window[] = GLFW.CreateWindow(window_width, window_height, "Julia OpenGL Shapes & Text Demo")
    if window == C_NULL
        GLFW.Terminate()
        error("Failed to create GLFW window")
        return -1
    end
    GLFW.MakeContextCurrent(window[])
    GLFW.ShowWindow(window[])

    # Get window size (in screen coordinates)
    window_size = GLFW.GetWindowSize(window[])

    # Get framebuffer size (in pixels)
    framebuffer_size = GLFW.GetFramebufferSize(window[])

    # Calculate scaling factor
    scale_x = framebuffer_size.width / window_size.width
    scale_y = framebuffer_size.height / window_size.height
    @info "DPI Scaling: $scale_x $scale_y"

    # Query and print OpenGL info
    @info "OpenGL Context Info:" create_context_info()

    # Enable blending for text/texture transparency
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    initialize_render_context()
    get_context().width = window_width
    get_context().height = window_height
    get_context().dpi_scaling = scale_x

    # Setup callbacks
    GLFW.SetFramebufferSizeCallback(window[], function (_, w, h)
        get_context().width = w
        get_context().height = h
    end)

    # Enable VSync
    GLFW.SwapInterval(1)

    # Initial projection matrix setup
    update_ortho_projection_matrix(window_width, window_height, scale_x)

    global terminate = function ()
        @info "Cleaning up resources..."
        cleanup_render_context()
        GLFW.DestroyWindow(window[])
        GLFW.Terminate()
        @info "Shutdown complete."
    end

    return window[]
end

function start_render_loop(render::Function; wait_for_events::Bool = false)
    frame_count::Int64 = 0
    while !GLFW.WindowShouldClose(window[])
        try
            frame_count += 1

            bg_color = 0.0f0 # Gray background
            glClearColor(bg_color, bg_color, bg_color, 1.0f0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)

            render()
            # --- End Frame ---
            GLFW.SwapBuffers(window[])
            if wait_for_events
                GLFW.WaitEvents()
            else
                GLFW.PollEvents()
            end
            gl_check_error("end of frame $frame_count") # Check for errors each frame
        catch e
            println("Error in main loop: ", e)
            showerror(stdout, e, catch_backtrace())
            println()
            sleep(0.5)
        end
    end
    terminate()
end

function julia_main()::Cint
    initialize()

    # --- Load Assets ---
    # Example: Load a texture (provide a path to an actual image file)
    # Create a dummy texture if no file exists
    test_texture_id = GLuint(0)
    try
        # IMPORTANT: Replace with the actual path to your image file
        test_texture_path = "testimage.jpg" # Example path
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
    #=
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
    =#
    # --- End Makie Plot ---

    # --- Main Loop ---
    frame_count::Int64 = 0
    last_frame_time = time() # Initialize time measurement before the loop
    #circle = create_circle(100f0)

    start_render_loop(function ()
        frame_count += 1
        current_frame_time = time() # Get time at the start of the frame processing
        delta_time = current_frame_time - last_frame_time
        last_frame_time = current_frame_time

        update_ortho_projection_matrix()
        #update_perspective_projection_matrix()

        # Demo drawing calls:
        # Draw a solid red rectangle
        save()
        fillcolor(rgba(255, 0, 0, 255))
        fillrect(50.0f0, 50.0f0, 100.0f0, 80.0f0)
        restore()

        # Draw a solid green rectangle
        save()
        fillcolor(rgba(0, 255, 0, 255))
        fillrect(200.0f0, 100.0f0, 50.0f0, 150.0f0)
        restore()

        save()
        translate(frame_count, frame_count)
        scale(sin(frame_count * 0.05) * 0.25 + 0.75)
        #draw_mesh(circle)
        #beginpath()
        fillcolor(rgba(255, 255, 255, 100))
        fillcircle(100)
        #fill()
        restore()

        # Draw the loaded texture (if available)
        if test_texture_id != 0
            # Full size
            drawimage(50.0f0, 200.0f0, 150.0f0, 150.0f0, get_context().font_texture)
            # Tinted blue and scaled
            save()
            translate(250.0f0, 250.0f0)
            scale(10)
            drawimage(0, 0, 0100.0f0, 100.0f0, test_texture_id)
            restore()
        end

        # Calculate FPS using delta_time (add epsilon to prevent division by zero)
        fps = round(Int, 1.0 / (delta_time + 1e-9))

        # Draw text (using the simplified font renderer)
        save()
        translate(50, 400)
        scale(2)
        fillcolor(rgba(255, 255, 0, 200))
        text("Hello Julia OpenGL!") # Yellow, scaled up
        restore()

        save()
        translate(10, 10)
        text("FPS: $fps") # White, normal size
        restore()

        save()
        translate(50, 450)
        scale(2)
        fillcolor(rgba(20, 200, 255, 255))
        text("0123456789 ASCII /?!") # Cyan
        restore()

        save()
        moveto(100, 100)
        lineto(200, 200)
        lineto(300, 200)
        lineto(350, 500)
        moveto(500, 200)
        lineto(500, 300)
        strokewidth(2)
        strokecolor(rgba(64, 128, 255, 255))
        stroke()
        restore()

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
    end)

    return 0
end

function test_scene_3d()
    initialize()

    frame_count::Int64 = 0

    test_texture = load_texture("./testimage.jpg")
    cube_mesh = create_cube(10.0f0)

    glEnable(GL_DEPTH_TEST)

    start_render_loop(function ()
        frame_count += 1
        save()
        update_perspective_projection_matrix()

        #translate(frame_count / 100, 0, -20)
        fillcolor(rgba(255, 255, 255, 255))
        fillrect(0, 0, 1, 1)
        drawimage(0, 0, 1, 1, test_texture)

        lookat(Float32[cos(frame_count / 100) * 30, sin(frame_count / 100) * 30, 0], Float32[0, 0, 0], Float32[0, 0, 1])

        save()
        translate(0, 0, 10)
        scale(0.5)
        draw_mesh(cube_mesh, test_texture)
        save()
        strokecolor(rgba(255, 0, 0, 255))
        #fillcolor(rgba(255, 0, 0, 255))
        #fillrect(0.0f0, 0.0f0, 100.0f0, 80.0f0)
        beginpath()
        moveto(100, 100)
        lineto(0, 0)
        lineto(-100, 0)
        stroke()
        restore()

        restore()

        save()
        translate(10, 0, 0)
        rotate(pi / 2)
        scale(0.5)
        draw_mesh(cube_mesh, test_texture)
        restore()

        #rotate(frame_count / 30.6, frame_count / 20, frame_count / 40)
        draw_mesh(cube_mesh, test_texture)

        restore()
    end)
end

export
    save,
    restore,
    moveto,
    lineto,
    beginpath,
    fill,
    stroke,
    strokecolor,
    fillcolor,
    rgba,
    strokewidth,
    text,
    translate,
    rotate,
    lookat,
    scale,
    drawimage,
    fillrect,
    fillcircle,
    rect,
    circle,
    Mesh,
    create_mesh,
    create_3d_mesh,
    update_mesh_vertices,
    draw_mesh,
    RenderContext,
    set_render_context,
    initialize_render_context,
    update_projection_matrix,
    load_texture,
    initialize,
    start_render_loop

end # module GraphicsTest

