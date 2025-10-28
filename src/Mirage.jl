module Mirage

import GLFW
using ModernGL
using Revise
using FileIO
using ImageIO
using Images

"""
    translate!(matrix::Matrix{T}, tx::Real, ty::Real, tz::Real = 0.0) where T

Applies a translation to a 4x4 transformation matrix in-place.

# Arguments
- `matrix`: The 4x4 transformation matrix to modify.
- `tx`: The translation amount along the x-axis.
- `ty`: The translation amount along the y-axis.
- `tz`: The translation amount along the z-axis (defaults to 0.0).

# Returns
The modified transformation matrix.
"""
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

"""
    rotate!(matrix::Matrix{T}, angle::Real) where T

Applies a 2D rotation around the Z-axis to a 4x4 transformation matrix in-place.

# Arguments
- `matrix`: The 4x4 transformation matrix to modify.
- `angle`: The rotation angle in radians.

# Returns
The modified transformation matrix.
"""
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

"""
    rotate!(matrix::Matrix{T}, angle::Real, axis::Vector{T}) where T

Applies a 3D rotation around a specified axis to a 4x4 transformation matrix in-place.

# Arguments
- `matrix`: The 4x4 transformation matrix to modify.
- `angle`: The rotation angle in radians.
- `axis`: A 3-element vector specifying the rotation axis.

# Throws
- `ArgumentError`: If the axis is not a 3-element vector or is a zero vector.

# Returns
The modified transformation matrix.
"""
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

"""
    scale!(matrix::Matrix{T}, sx::Real, sy::Real, sz::Real = 1.0) where T

Applies a scaling transformation to a 4x4 transformation matrix in-place.

# Arguments
- `matrix`: The 4x4 transformation matrix to modify.
- `sx`: The scaling factor along the x-axis.
- `sy`: The scaling factor along the y-axis.
- `sz`: The scaling factor along the z-axis (defaults to 1.0).

# Returns
The modified transformation matrix.
"""
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

"""
    gl_gen_one(gl_gen_fn)

Generates a single OpenGL object (buffer, vertex array, or texture) using the provided generation function.

# Arguments
- `gl_gen_fn`: The OpenGL generation function (e.g., `glGenBuffers`, `glGenVertexArrays`, `glGenTextures`).

# Returns
The ID of the generated OpenGL object.
"""
function gl_gen_one(gl_gen_fn)
    id = GLuint[0]
    gl_gen_fn(1, id)
    gl_check_error("generating a buffer, array, or texture")
    id[]
end

"""
    gl_gen_buffer()

Generates a single OpenGL buffer object.

# Returns
The ID of the generated buffer object.
"""
gl_gen_buffer() = gl_gen_one(glGenBuffers)

"""
    gl_gen_vertex_array()

Generates a single OpenGL vertex array object.

# Returns
The ID of the generated vertex array object.
"""
gl_gen_vertex_array() = gl_gen_one(glGenVertexArrays)

"""
    gl_gen_texture()

Generates a single OpenGL texture object.

# Returns
The ID of the generated texture object.
"""
gl_gen_texture() = gl_gen_one(glGenTextures)

"""
    get_info_log(obj::GLuint)

Retrieves the info log for an OpenGL shader or program object.

# Arguments
- `obj`: The ID of the shader or program object.

# Returns
A string containing the info log, or an empty string if no log is available.
"""
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

"""
    validate_shader(shader)

Checks if an OpenGL shader compilation was successful.

# Arguments
- `shader`: The ID of the shader object.

# Returns
`true` if the shader compiled successfully, `false` otherwise.
"""
function validate_shader(shader)
    success = GLint[0]
    glGetShaderiv(shader, GL_COMPILE_STATUS, success)
    success[] == GL_TRUE
end

"""
    gl_error_message()

Retrieves the current OpenGL error message as a string.

# Returns
A string describing the OpenGL error, or an empty string if no error occurred.
"""
function gl_error_message()
    err = glGetError()
    err == GL_NO_ERROR ? "" :
        err == GL_INVALID_ENUM ? "GL_INVALID_ENUM" :
        err == GL_INVALID_VALUE ? "GL_INVALID_VALUE" :
        err == GL_INVALID_OPERATION ? "GL_INVALID_OPERATION" :
        err == GL_INVALID_FRAMEBUFFER_OPERATION ? "GL_INVALID_FRAMEBUFFER_OPERATION" :
        err == GL_OUT_OF_MEMORY ? "GL_OUT_OF_MEMORY" : "Unknown OpenGL error code $err."
end

"""
    gl_check_error(action_name="")

Checks for OpenGL errors and throws an `error` if one is found.

# Arguments
- `action_name`: An optional string describing the action being performed when the error check occurs.

# Throws
- `error`: If an OpenGL error is detected.
"""
function gl_check_error(action_name="")
    message = gl_error_message()
    if length(message) > 0
        error("OpenGL Error", isempty(action_name) ? "" : " during $action_name", ": ", message)
    end
end

"""
    create_shader(source, typ)

Creates and compiles an OpenGL shader from source.

# Arguments
- `source`: A string containing the shader source code.
- `typ`: The type of shader to create (e.g., `GL_VERTEX_SHADER`, `GL_FRAGMENT_SHADER`).

# Returns
The ID of the compiled shader.

# Throws
- `error`: If shader creation or compilation fails.
"""
function create_shader(source, typ)
    shader::GLuint = glCreateShader(typ)
    if shader == 0
        error("Error creating shader: ", gl_error_message())
    end
    glShaderSource(shader, 1, Ref(pointer(source)), C_NULL)
    glCompileShader(shader)
    !validate_shader(shader) && error("Shader compilation error: ", get_info_log(shader))
    return shader
end

"""
    create_shader_program(vertex_shader::GLuint, fragment_shader::GLuint)::GLuint

Creates an OpenGL shader program by linking a vertex and a fragment shader.

# Arguments
- `vertex_shader`: The ID of the vertex shader.
- `fragment_shader`: The ID of the fragment shader.

# Returns
The ID of the linked shader program.

# Throws
- `error`: If program creation or linking fails.
"""
function create_shader_program(vertex_shader::GLuint, fragment_shader::GLuint)::GLuint
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

"""
    create_shader_program(vertex_shader::String, fragment_shader::String)::ShaderInfo

Creates an OpenGL shader program from vertex and fragment shader source strings.

# Arguments
- `vertex_shader`: A string containing the vertex shader source code.
- `fragment_shader`: A string containing the fragment shader source code.

# Returns
A `ShaderInfo` object containing the program ID and a dictionary for uniform locations.
"""
function create_shader_program(vertex_shader::String, fragment_shader::String)::ShaderInfo
    vertex_shader_id = create_shader(vertex_shader, GL_VERTEX_SHADER)
    fragment_shader_id = create_shader(fragment_shader, GL_FRAGMENT_SHADER)
    shader = create_shader_program(vertex_shader_id, fragment_shader_id)
    glDeleteShader(vertex_shader_id)
    glDeleteShader(fragment_shader_id)
    return ShaderInfo(shader, Dict{String, GLint}())
end

global glsl_version = ""
global immediate_mesh = nothing
global immediate_mesh_3d = nothing

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

"""
    initialize_shader_uniform!(shader::ShaderInfo, uniform_name::String)

Retrieves the location of a uniform variable in a shader program and stores it in the `ShaderInfo` object.

# Arguments
- `shader`: The `ShaderInfo` object representing the shader program.
- `uniform_name`: The name of the uniform variable.
"""
function initialize_shader_uniform!(shader::ShaderInfo, uniform_name::String)
    shader.uniform_locations[uniform_name] = glGetUniformLocation(shader.program_id, uniform_name)
end

"""
    create_context_info()

Retrieves and processes OpenGL context information, including GLSL version, OpenGL version, vendor, and renderer.

# Returns
A dictionary containing OpenGL context details.

# Throws
- `error`: If the GLSL or OpenGL version strings are in an unexpected format.
"""
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
"""
    ortho(left::Float32, right::Float32, bottom::Float32, top::Float32, zNear::Float32 = -1.0f0, zFar::Float32 = 1.0f0)::Matrix{Float32}

Creates an orthographic projection matrix.

# Arguments
- `left`: The x-coordinate of the left vertical clipping plane.
- `right`: The x-coordinate of the right vertical clipping plane.
- `bottom`: The y-coordinate of the bottom horizontal clipping plane.
- `top`: The y-coordinate of the top horizontal clipping plane.
- `zNear`: The distance to the near clipping plane (defaults to -1.0f0).
- `zFar`: The distance to the far clipping plane (defaults to 1.0f0).

# Returns
A 4x4 orthographic projection matrix.
"""
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

"""
    perspective(fov::Float32, aspectRatio::Float32, near::Float32, far::Float32)::Matrix{Float32}

Creates a perspective projection matrix.

# Arguments
- `fov`: The field of view in radians.
- `aspectRatio`: The aspect ratio of the viewport (width / height).
- `near`: The distance to the near clipping plane.
- `far`: The distance to the far clipping plane.

# Returns
A 4x4 perspective projection matrix.
"""
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

"""
    view(position, target, up = [0, 0, 1])

Creates a view matrix (camera matrix) that transforms world coordinates to view coordinates.

# Arguments
- `position`: The position of the camera in world space.
- `target`: The point in world space that the camera is looking at.
- `up`: The up direction of the camera (defaults to `[0, 0, 1]`).

# Returns
A 4x4 view matrix.
"""
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

"""
    normalize(v::Vector{Float32})::Vector{Float32}

Normalizes a 3-element Float32 vector.

# Arguments
- `v`: The input vector.

# Returns
The normalized vector.
"""
function normalize(v::Vector{Float32})::Vector{Float32}
    len = sqrt(sum(v .^ 2))
    return len > 0.0f0 ? v ./ len : v
end

"""
    cross(a::Vector{Float32}, b::Vector{Float32})::Vector{Float32}

Computes the cross product of two 3-element Float32 vectors.

# Arguments
- `a`: The first vector.
- `b`: The second vector.

# Returns
The cross product vector.
"""
function cross(a::Vector{Float32}, b::Vector{Float32})::Vector{Float32}
    return Float32[
        a[2] * b[3] - a[3] * b[2],
        a[3] * b[1] - a[1] * b[3],
        a[1] * b[2] - a[2] * b[1]
    ]
end

"""
    dot(a::Vector{Float32}, b::Vector{Float32})::Float32

Computes the dot product of two 3-element Float32 vectors.

# Arguments
- `a`: The first vector.
- `b`: The second vector.

# Returns
The dot product (a scalar value).
"""
function dot(a::Vector{Float32}, b::Vector{Float32})::Float32
    return sum(a .* b)
end

"""
    update_ortho_projection_matrix(width=get_context().width,
                                   height=get_context().height,
                                   dpi_scaling=get_context().dpi_scaling)

Updates the orthographic projection matrix based on the current context's width, height, and DPI scaling.
Also sets the OpenGL viewport.

# Arguments
- `width`: The width of the viewport (defaults to `get_context().width`).
- `height`: The height of the viewport (defaults to `get_context().height`).
- `dpi_scaling`: The DPI scaling factor (defaults to `get_context().dpi_scaling`).
"""
function update_ortho_projection_matrix(width=get_context().width,
                                        height=get_context().height,
                                        dpi_scaling=get_context().dpi_scaling)
    # Map pixel coords (0, width) -> (-1, 1) and (0, height) -> (1, -1)
    get_state().projection = ortho(0.0f0, Float32(width / dpi_scaling), Float32(height / dpi_scaling), 0.0f0)
    glViewport(0, 0, width, height)
end

"""
    update_perspective_projection_matrix(width=get_context().width,
                                         height=get_context().height,
                                         dpi_scaling=get_context().dpi_scaling;
                                         near = 0.01,
                                         far = 10_000)

Updates the perspective projection matrix based on the current context's width, height, and DPI scaling.
Also sets the OpenGL viewport.

# Arguments
- `width`: The width of the viewport (defaults to `get_context().width`).
- `height`: The height of the viewport (defaults to `get_context().height`).
- `dpi_scaling`: The DPI scaling factor (defaults to `get_context().dpi_scaling`).
- `near`: The distance to the near clipping plane (defaults to 0.01).
- `far`: The distance to the far clipping plane (defaults to 10_000).
"""
function update_perspective_projection_matrix(width=get_context().width,
                                              height=get_context().height,
                                              dpi_scaling=get_context().dpi_scaling;
                                              near = 0.01,
                                              far = 10_000)
    get_state().projection = perspective(Float32(pi / 4), Float32(width / height), Float32(near), Float32(far))
    glViewport(0, 0, width, height)
end

"""
    load_texture(filepath::String)::GLuint

Loads an image from the specified filepath and creates an OpenGL texture from it.

# Arguments
- `filepath`: The path to the image file.

# Returns
The ID of the created OpenGL texture.

# Throws
- `error`: If the image cannot be loaded.
"""
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

"""
    load_texture(img_rgba::Matrix{Images.RGBA{Images.N0f8}})::GLuint

Creates an OpenGL texture from a given RGBA image matrix.

# Arguments
- `img_rgba`: A matrix of `RGBA{N0f8}` representing the image data.

# Returns
The ID of the created OpenGL texture.
"""
function load_texture(img_rgba::Matrix{Images.RGBA{Images.N0f8}})::GLuint
    tex_height, tex_width = size(img_rgba)

    output_img = permutedims(img_rgba[end:-1:1, :], (2, 1))

    output_bytes = vec(reinterpret(UInt8, output_img))

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

mutable struct Canvas
    fbo::GLuint
    texture::GLuint
    rbo::GLuint # Renderbuffer Object for depth/stencil
    width::Int
    height::Int
end

"""
    create_canvas(width::Int, height::Int)

Creates a new `Canvas` object, which encapsulates an OpenGL framebuffer, texture, and renderbuffer for offscreen rendering.

# Arguments
- `width`: The width of the canvas in pixels.
- `height`: The height of the canvas in pixels.

# Returns
A `Canvas` object.

# Throws
- `@error`: If the framebuffer is not complete after creation.
"""
function create_canvas(width::Int, height::Int)
    # Generate Framebuffer
    fbo = gl_gen_one(glGenFramebuffers)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    gl_check_error("binding canvas FBO")

    # Create Texture Attachment
    texture = gl_gen_texture()
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, C_NULL)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
    gl_check_error("attaching canvas texture")

    # Create Renderbuffer for Depth/Stencil
    rbo = gl_gen_one(glGenRenderbuffers)
    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo)
    gl_check_error("attaching canvas renderbuffer")

    # Finalize and check status
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE
        @error "Framebuffer is not complete!"
    end

    # Unbind to return to default state
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    return Canvas(fbo, texture, rbo, width, height)
end

"""
    set_canvas(canvas::Canvas)

Sets the render target to the specified canvas. All subsequent drawing
commands will render to this canvas.

# Arguments
- `canvas`: The `Canvas` object to set as the current render target.
"""
function set_canvas(canvas::Canvas)
    glBindFramebuffer(GL_FRAMEBUFFER, canvas.fbo)
    glViewport(0, 0, canvas.width, canvas.height)
    get_context().context_stack = [ContextState()]
end

"""
    set_canvas()

Resets the render target to the main window. All subsequent drawing
commands will render to the main window.
"""
function set_canvas()
    ctx = get_context()
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glViewport(0, 0, ctx.width, ctx.height)
    get_context().context_stack = [ContextState()]
end

"""
    resize!(canvas::Canvas, width::Int, height::Int)

Resizes the canvas and its underlying texture and renderbuffer objects.

This is useful if the canvas needs to match a new window size or if a
different resolution is required for rendering effects.

# Arguments
- `canvas`: The `Canvas` object to resize.
- `width`: The new width of the canvas in pixels.
- `height`: The new height of the canvas in pixels.

# Returns
The modified `Canvas` object.
"""
function resize!(canvas::Canvas, width::Int, height::Int)
    # Ensure dimensions are valid before proceeding
    @assert width > 0 && height > 0 "Canvas dimensions must be positive"

    canvas.width = width
    canvas.height = height

    # Resize the texture attachment
    glBindTexture(GL_TEXTURE_2D, canvas.texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, C_NULL)
    gl_check_error("resizing canvas texture")

    # Resize the renderbuffer attachment for depth and stencil
    glBindRenderbuffer(GL_RENDERBUFFER, canvas.rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height)
    gl_check_error("resizing canvas renderbuffer")

    # Unbind objects to return to a clean state
    glBindTexture(GL_TEXTURE_2D, 0)
    glBindRenderbuffer(GL_RENDERBUFFER, 0)

    return canvas
end

"""
    destroy!(canvas::Canvas)

Frees the GPU resources associated with a Canvas.

This includes the framebuffer object (FBO), the color texture, and the
depth/stencil renderbuffer. It is essential to call this function when the
canvas is no longer needed to prevent memory leaks on the GPU.

# Arguments
- `canvas`: The `Canvas` object to destroy.
"""
function destroy!(canvas::Canvas)
    # Ensure we don't try to delete already-deleted objects
    if canvas.fbo == 0 && canvas.texture == 0 && canvas.rbo == 0
        @warn "Canvas has already been destroyed."
        return
    end

    glDeleteFramebuffers(1, [canvas.fbo])
    gl_check_error("deleting canvas FBO")

    glDeleteTextures(1, [canvas.texture])
    gl_check_error("deleting canvas texture")

    glDeleteRenderbuffers(1, [canvas.rbo])
    gl_check_error("deleting canvas RBO")

    # Set IDs to 0 to indicate that the resources have been freed
    canvas.fbo = 0
    canvas.texture = 0
    canvas.rbo = 0
end

@kwdef mutable struct ContextState
    transform::Matrix{Float32} = Float32[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
    view::Matrix{Float32} = Float32[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
    projection::Matrix{Float32} = ortho(0f0, 800f0, 600f0, 0f0)
    fill_color::Tuple{Float32, Float32, Float32, Float32} = (1, 1, 1, 1)
    stroke_color::Tuple{Float32, Float32, Float32, Float32} = (0, 0, 0, 1)
    stroke_width::Float32 = 1
    paths::Vector{Vector{Tuple{Float32, Float32, Float32}}} = [[]]
    current_path_index::Int = 1
end

"""
    clone(x::ContextState)

Creates a deep copy of a `ContextState` object.

# Arguments
- `x`: The `ContextState` object to clone.

# Returns
A new `ContextState` object with identical, but independent, field values.
"""
function clone(x::ContextState)
    fields = fieldnames(ContextState)
    kwargs = Dict{Symbol, Any}(field => deepcopy(getfield(x, field)) for field in fields)
    return ContextState(; kwargs...)
end

include("./default_font.jl")

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

    """
    RenderContext()

Constructs a new `RenderContext` object, initializing OpenGL shaders, textures, and context state.

# Returns
A new `RenderContext` instance.
"""
    function RenderContext()::RenderContext
        shader = create_shader_program(
            texture_vertex_shader_source,
            texture_fragment_shader_source
        )

        initialize_shader_uniform!(shader, "projection")
        initialize_shader_uniform!(shader, "view")
        initialize_shader_uniform!(shader, "model")
        initialize_shader_uniform!(shader, "textureSampler")
        initialize_shader_uniform!(shader, "color")

        blank_texture = gl_gen_texture()
        #font_texture = load_texture("./ascii_font_atlas.png")
        font_texture = load_texture(map(x -> x == 1 ? RGBA(1, 1, 1, 1) : RGBA(0, 0, 0, 0), default_font))

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
            [ContextState()],
            800, 600,
            1.0
        )
    end
end

const render_context = Ref{RenderContext}()

"""
    cleanup_render_context(ctx::RenderContext = get_context())

Cleans up OpenGL resources associated with a `RenderContext`, including shader programs and textures.

# Arguments
- `ctx`: The `RenderContext` to clean up (defaults to the current context).
"""
function cleanup_render_context(ctx::RenderContext = get_context())
    glDeleteProgram(ctx.shader.program_id)
    glDeleteTextures(1, [ctx.blank_texture])
    glDeleteTextures(1, [ctx.font_texture])
    global immediate_mesh = nothing
    global immediate_mesh_3d = nothing
end

"""
    save()

Pushes a copy of the current `ContextState` onto the context stack, effectively saving the current drawing state.
"""
save() = push!(get_context().context_stack, clone(get_context().context_stack[end]))

"""
    restore()

Pops the last saved `ContextState` from the context stack, restoring the previous drawing state.
"""
restore() = pop!(get_context().context_stack)

"""
    get_context()

Retrieves the global `RenderContext` instance.

# Returns
The global `RenderContext`.
"""
get_context() = render_context[]

"""
    get_state()

Retrieves the current `ContextState` from the top of the context stack.

# Returns
The current `ContextState`.
"""
get_state() = get_context().context_stack[end]

"""
    translate(dx::Number, dy::Number, dz::Number = 0)

Applies a translation to the current transformation matrix in the `ContextState`.

# Arguments
- `dx`: The translation amount along the x-axis.
- `dy`: The translation amount along the y-axis.
- `dz`: The translation amount along the z-axis (defaults to 0).
"""
function translate(dx::Number, dy::Number, dz::Number = 0)
    translate!(get_state().transform, dx, dy, dz)
end

"""
    scale(dx::Number, dy::Number, dz::Number = 1)

Applies a scaling transformation to the current transformation matrix in the `ContextState`.

# Arguments
- `dx`: The scaling factor along the x-axis.
- `dy`: The scaling factor along the y-axis.
- `dz`: The scaling factor along the z-axis (defaults to 1).
"""
function scale(dx::Number, dy::Number, dz::Number = 1)
    scale!(get_state().transform, dx, dy, dz)
end
"""
    scale(n::Number)

Applies a uniform scaling transformation to the current transformation matrix in the `ContextState`.

# Arguments
- `n`: The uniform scaling factor for all axes.
"""
scale(n::Number) = scale(n, n, n)

"""
    rotate(angle::Number)

Applies a 2D rotation around the Z-axis to the current transformation matrix in the `ContextState`.

# Arguments
- `angle`: The rotation angle in radians.
"""
function rotate(angle::Number)
    rotate!(get_state().transform, angle)
end

"""
    rotate(x::Number, y::Number, z::Number)

Applies rotations around the X, Y, and Z axes sequentially to the current transformation matrix in the `ContextState`.

# Arguments
- `x`: The rotation angle around the X-axis in radians.
- `y`: The rotation angle around the Y-axis in radians.
- `z`: The rotation angle around the Z-axis in radians.
"""
function rotate(x::Number, y::Number, z::Number)
    rotate!(get_state().transform, x, Float32[1, 0, 0])
    rotate!(get_state().transform, y, Float32[0, 1, 0])
    rotate!(get_state().transform, z, Float32[0, 0, 1])
end

"""
    lookat(args...)

Sets the view matrix in the current `ContextState` using the `view` function.

# Arguments
- `args...`: Arguments passed directly to the `view` function (e.g., `position`, `target`, `up`).
"""
function lookat(args...)
    get_state().view = view(args...)
end

"""
    beginpath()

Clears the current paths and starts a new path in the `ContextState`.
"""
function beginpath()
    get_state().paths = [[]]
    get_state().current_path_index = 1
end

"""
    moveto(x::Number, y::Number, z::Number = 0.0)

Moves the current drawing position to the specified coordinates, starting a new subpath if the current one is not empty.

# Arguments
- `x`: The x-coordinate to move to.
- `y`: The y-coordinate to move to.
- `z`: The z-coordinate to move to (defaults to 0.0).
"""
function moveto(x::Number, y::Number, z::Number = 0.0)
    state = get_state()
    if !isempty(state.paths[state.current_path_index])
        push!(state.paths, [])
        state.current_path_index += 1
    end
    push!(state.paths[state.current_path_index], (Float32(x), Float32(y), Float32(z)))
end

"""
    lineto(x::Number, y::Number, z::Number = 0.0)

Adds a line segment from the current drawing position to the specified coordinates.

# Arguments
- `x`: The x-coordinate to draw the line to.
- `y`: The y-coordinate to draw the line to.
- `z`: The z-coordinate to draw the line to (defaults to 0.0).
"""
function lineto(x::Number, y::Number, z::Number = 0.0)
    push!(get_state().paths[get_state().current_path_index], (Float32(x), Float32(y), Float32(z)))
end

"""
    closepath()

Closes the current path by adding a line segment from the current point to the starting point of the subpath.
"""
function closepath()
    state = get_state()
    current_path = state.paths[state.current_path_index]
    if !isempty(current_path)
        push!(current_path, current_path[1])
    end
end

"""
    fillcolor(tuple::Tuple{Number, Number, Number})

Sets the fill color for subsequent drawing operations using an RGB tuple. The alpha component is set to 1.

# Arguments
- `tuple`: A tuple `(r, g, b)` where `r`, `g`, `b` are color components (0-1 or 0-255).
"""
function fillcolor(tuple::Tuple{Number, Number, Number})
    get_state().fill_color = (
        Float32(tuple[1]),
        Float32(tuple[2]),
        Float32(tuple[3]),
        Float32(1)
    )
end

"""
    fillcolor(tuple::Tuple{Number, Number, Number, Number})

Sets the fill color for subsequent drawing operations using an RGBA tuple.

# Arguments
- `tuple`: A tuple `(r, g, b, a)` where `r`, `g`, `b`, `a` are color components (0-1 or 0-255).
"""
function fillcolor(tuple::Tuple{Number, Number, Number, Number})
    get_state().fill_color = (
        Float32(tuple[1]),
        Float32(tuple[2]),
        Float32(tuple[3]),
        Float32(tuple[4])
    )
end

"""
    strokecolor(tuple::Tuple{Number, Number, Number})

Sets the stroke color for subsequent drawing operations using an RGB tuple. The alpha component is set to 1.

# Arguments
- `tuple`: A tuple `(r, g, b)` where `r`, `g`, `b` are color components (0-1 or 0-255).
"""
function strokecolor(tuple::Tuple{Number, Number, Number})
    get_state().stroke_color = (
        Float32(tuple[1]),
        Float32(tuple[2]),
        Float32(tuple[3]),
        Float32(1)
    )
end

"""
    strokecolor(tuple::Tuple{Number, Number, Number, Number})

Sets the stroke color for subsequent drawing operations using an RGBA tuple.

# Arguments
- `tuple`: A tuple `(r, g, b, a)` where `r`, `g`, `b`, `a` are color components (0-1 or 0-255).
"""
function strokecolor(tuple::Tuple{Number, Number, Number, Number})
    get_state().stroke_color = (
        Float32(tuple[1]),
        Float32(tuple[2]),
        Float32(tuple[3]),
        Float32(tuple[4])
    )
end

"""
    rgba(r::Int, g::Int, b::Int, a::Int = 255)::Tuple{Float32, Float32, Float32, Float32}

Creates an RGBA color tuple with Float32 components (0.0-1.0) from Int components (0-255).

# Arguments
- `r`: Red component (0-255).
- `g`: Green component (0-255).
- `b`: Blue component (0-255).
- `a`: Alpha component (0-255, defaults to 255).

# Returns
A `Tuple{Float32, Float32, Float32, Float32}` representing the RGBA color.
"""
function rgba(r::Int, g::Int, b::Int, a::Int = 255)::Tuple{Float32, Float32, Float32, Float32}
    return (r / 255, g / 255, b / 255, a / 255)
end

"""
    strokewidth(w::Number)

Sets the stroke width for subsequent drawing operations.

# Arguments
- `w`: The desired stroke width.
"""
function strokewidth(w::Number)
    get_state().stroke_width = w
end

"""
    stroke()

Draws the currently defined paths as stroked lines using the current stroke color and width.
"""
function stroke()
    immediate_mesh = get_immediate_mesh_3d()
    state::ContextState = get_state()
    vertices::Vector{Float32} = Float32[]

    for path in state.paths
        if length(path) < 2
            continue
        end

        for i in 1:(length(path) - 1)
            (x1, y1, z1) = path[i]
            (x2, y2, z2) = path[i + 1]

            angle::Float64 = atan(y2 - y1, x2 - x1)
            dx::Float32 = cos(angle + pi / 2) * state.stroke_width / 2
            dy::Float32 = sin(angle + pi / 2) * state.stroke_width / 2

            append!(vertices, Float32[
                x1 + dx, y1 + dy, z1, 0.0, 0.0,
                x1 - dx, y1 - dy, z1, 0.0, 1.0,
                x2 + dx, y2 + dy, z2, 1.0, 1.0,
                x1 - dx, y1 - dy, z1, 0.0, 1.0,
                x2 - dx, y2 - dy, z2, 1.0, 0.0,
                x2 + dx, y2 + dy, z2, 1.0, 1.0,
            ])
        end
    end

    if !isempty(vertices)
        update_mesh_vertices!(immediate_mesh, vertices)
        draw_mesh(immediate_mesh, get_context().blank_texture, [state.stroke_color...])
    end
end

"""
    fill()

Fills the currently defined paths using the current fill color.
"""
function fill()
    immediate_mesh = get_immediate_mesh()
    state::ContextState = get_state()

    for path in state.paths
        if length(path) < 3
            continue
        end

        # Simple triangulation using the first vertex as the center
        center_x, center_y = path[1]
        vertices = Float32[]

        for i in 2:(length(path) - 1)
            x1, y1 = path[i]
            x2, y2 = path[i + 1]

            append!(vertices, Float32[
                center_x, center_y, 0.5, 0.5, # Center vertex
                x1, y1, 0.0, 0.0,             # First vertex on edge
                x2, y2, 1.0, 0.0              # Second vertex on edge
            ])
        end

        if !isempty(vertices)
            update_mesh_vertices!(immediate_mesh, vertices)
            draw_mesh(immediate_mesh, get_context().blank_texture, [state.fill_color...])
        end
    end
end

"""
    rect(x::Number, y::Number, w::Number, h::Number)

Defines a rectangular path.

# Arguments
- `x`: The x-coordinate of the top-left corner of the rectangle.
- `y`: The y-coordinate of the top-left corner of the rectangle.
- `w`: The width of the rectangle.
- `h`: The height of the rectangle.
"""
function rect(x::Number, y::Number, w::Number, h::Number)
    beginpath()
    moveto(x, y)
    lineto(x + w, y)
    lineto(x + w, y + h)
    lineto(x, y + h)
    closepath()
end

"""
    circle(r::Number, x::Number = 0, y::Number = 0, segments::Int = 32)

Defines a circular path.

# Arguments
- `r`: The radius of the circle.
- `x`: The x-coordinate of the center of the circle (defaults to 0).
- `y`: The y-coordinate of the center of the circle (defaults to 0).
- `segments`: The number of line segments used to approximate the circle (defaults to 32).
"""
function circle(r::Number, x::Number = 0, y::Number = 0, segments::Int = 32)
    for i in 0:segments
        angle::Float32 = 2.0f0 * pi * (i - 1) / segments
        next_angle::Float32 = 2.0f0 * pi * i / segments
        x::Float32 = r * cos(angle)
        y::Float32 = r * sin(angle)
        if i == 0
            moveto(x, y)
        else
            lineto(x, y)
        end
    end
end

"""
    fillrect(x::Number, y::Number, w::Number, h::Number)

Draws a filled rectangle using the current fill color.

# Arguments
- `x`: The x-coordinate of the top-left corner of the rectangle.
- `y`: The y-coordinate of the top-left corner of the rectangle.
- `w`: The width of the rectangle.
- `h`: The height of the rectangle.
"""
function fillrect(x::Number, y::Number, w::Number, h::Number)
    drawimage(x, y, w, h, get_context().blank_texture)
end

"""
    fillcircle(radius::Number, x::Number = 0, y::Number = 0, segments::Int = 32)

Draws a filled circle using the current fill color.

# Arguments
- `radius`: The radius of the circle.
- `x`: The x-coordinate of the center of the circle (defaults to 0).
- `y`: The y-coordinate of the center of the circle (defaults to 0).
- `segments`: The number of line segments used to approximate the circle (defaults to 32).
"""
function fillcircle(radius::Number, x::Number = 0, y::Number = 0, segments::Int = 32)
    # Create triangles by connecting center to each pair of consecutive vertices
    # This creates a fan-like triangulation
    vertices = Float32[]

    for i in 1:segments
        angle::Float32 = 2.0f0 * pi * (i - 1) / segments
        next_angle::Float32 = 2.0f0 * pi * i / segments

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

"""
    drawimage(x::Number,
              y::Number,
              w::Number,
              h::Number,
              texture_id::GLuint)

Draws a textured rectangle.

# Arguments
- `x`: The x-coordinate of the top-left corner of the rectangle.
- `y`: The y-coordinate of the top-left corner of the rectangle.
- `w`: The width of the rectangle.
- `h`: The height of the rectangle.
- `texture_id`: The OpenGL ID of the texture to draw.
"""
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
"""
    text(text::String)

Draws a string of text using the loaded font atlas.

# Arguments
- `text`: The string to draw.
"""
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

"""
    get_immediate_mesh()

Retrieves or creates the global 2D immediate mode mesh for drawing.

# Returns
The `Mesh` object for 2D immediate mode drawing.
"""
function get_immediate_mesh()
    global immediate_mesh
    if immediate_mesh == nothing || immediate_mesh.vao == 0
        @debug "Creating new immediate mesh"
        immediate_mesh = create_mesh()
    end
    return immediate_mesh
end

"""
    get_immediate_mesh_3d()

Retrieves or creates the global 3D immediate mode mesh for drawing.

# Returns
The `Mesh` object for 3D immediate mode drawing.
"""
function get_immediate_mesh_3d()
    global immediate_mesh_3d
    if immediate_mesh_3d == nothing || immediate_mesh_3d.vao == 0
        @debug "Creating new immediate mesh 3D"
        immediate_mesh_3d = create_3d_mesh()
    end
    return immediate_mesh_3d
end

include("./meshes.jl")

const window = Ref{GLFW.Window}()

"""
    set_render_context(ctx::RenderContext)

Sets the global `RenderContext`.

# Arguments
- `ctx`: The `RenderContext` to set as global.
"""
function set_render_context(ctx::RenderContext)
    render_context[] = ctx
end

"""
    initialize_render_context()

Initializes the global `RenderContext`.
"""
function initialize_render_context()
    set_render_context(RenderContext())
end

"""
    clear()

Clears the color, depth, and stencil buffers of the current OpenGL context.
"""
function clear()
    glClearColor(0.0f0, 0.0f0, 0.0f0, 0.0f0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
end

"""
    initialize(;window_width::Int = 800, window_height::Int = 600)

Initializes GLFW and OpenGL, creates a window, and sets up the rendering context.

# Arguments
- `window_width`: The desired width of the window (defaults to 800).
- `window_height`: The desired height of the window (defaults to 600).

# Returns
The GLFW window object.

# Throws
- `error`: If GLFW initialization or window creation fails.
"""
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
    GLFW.WindowHint(GLFW.SAMPLES, 4)

    # Create a windowed mode window and its OpenGL context
    window[] = GLFW.CreateWindow(window_width, window_height, "Mirage")
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
    glEnable(GL_MULTISAMPLE)

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
    update_ortho_projection_matrix(window_width, window_height, 1.0)

    global terminate = function ()
        @info "Cleaning up resources..."
        cleanup_render_context()
        GLFW.DestroyWindow(window[])
        GLFW.Terminate()
        @info "Shutdown complete."
    end

    return window[]
end

"""
    start_render_loop(render::Function; wait_for_events::Bool = false)

Starts the main render loop, continuously calling the `render` function until the window is closed.

# Arguments
- `render`: A function that contains the drawing logic for each frame.
- `wait_for_events`: If `true`, the loop will wait for events, otherwise it will poll for events (defaults to `false`).
"""
function start_render_loop(render::Function; wait_for_events::Bool = false)
    frame_count::Int64 = 0
    while !GLFW.WindowShouldClose(window[])
        try
            frame_count += 1
            clear()
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

"""
    test_scene_2d()

Runs a 2D test scene demonstrating various drawing functionalities like rectangles, circles, textures, and text.
"""
function test_scene_2d()
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
        #lookat(Float32[0, 0, -1000], Float32[0, 0, 0], Float32[0, -1, 0])

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
        circle(100)
        fill()
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

        save()
        beginpath()
        strokewidth(5)
        strokecolor(rgba(255, 255, 0, 255))
        moveto(400, 400)
        lineto(500, 400)
        lineto(500, 500)
        lineto(400, 500)
        closepath()
        fillcolor(rgba(0, 0, 255, 100))
        fill()
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
end

"""
    test_scene_3d()

Runs a 3D test scene demonstrating various 3D drawing functionalities and camera controls.
"""
function test_scene_3d()
    initialize()

    frame_count::Int64 = 0

    canvas = create_canvas(24, 24)
    cube_mesh = create_cube(10.0f0)
    sphere_mesh = create_uv_sphere(4.0f0)
    obj_mesh = load_obj_mesh("cube.obj")

    set_canvas(canvas)
    clear()
    update_ortho_projection_matrix(canvas.width, canvas.height, 1.0)
    save()
    fillcolor(rgba(0, 0, 50, 255))
    fillrect(0, 0, canvas.width, canvas.height)
    translate(4, 4)
    fillcolor(rgba(255, 255, 255, 255))
    text(":)")
    restore()
    set_canvas()

    glEnable(GL_DEPTH_TEST)

    start_render_loop(function ()
        frame_count += 1
        save()
        update_perspective_projection_matrix()

        fillcolor(rgba(255, 255, 255, 255))
        fillrect(0, 0, 1, 1)

        lookat(Float32[cos(frame_count / 100) * 30, sin(frame_count / 100) * 30, 0], Float32[0, 0, 0], Float32[0, 0, 1])

        save()
        translate(0, 0, 10)
        scale(0.5)
        draw_mesh(cube_mesh, canvas.texture)
        save()
        strokecolor(rgba(255, 0, 0, 255))
        beginpath()
        moveto(100, 100)
        lineto(0, 0)
        lineto(-100, 0, -50)
        stroke()
        restore()

        restore()

        save()
        translate(10, 0, 0)
        rotate(pi / 2)
        scale(0.5)
        draw_mesh(obj_mesh, canvas.texture)
        restore()

        rotate(frame_count / 30.6, frame_count / 20, frame_count / 40)
        draw_mesh(sphere_mesh, canvas.texture)

        restore()
    end)

    destroy!(canvas)
    destroy_mesh(obj_mesh)
end

export
    save,
    restore,
    moveto,
    lineto,
    closepath,
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
    start_render_loop,
    create_canvas,
    set_canvas,
    resize!,
    destroy!,
    clear,
    load_obj_mesh

end # module Mirage
