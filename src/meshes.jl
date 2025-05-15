# Represents a single vertex attribute (position, normal, etc.)
struct VertexAttribute
    location::Int      # Shader location
    size::Int          # Number of components (e.g., 3 for vec3)
    type::GLenum       # GL_FLOAT, etc.
    normalized::Bool   # Whether to normalize fixed-point data
    offset::Int        # Byte offset within vertex
end

# Simplified mesh with just one VBO and no EBO
mutable struct Mesh
    vao::GLuint
    vbo::GLuint
    vertex_count::Int
    draw_mode::GLenum  # GL_TRIANGLES, etc.
    stride::Int        # Total bytes per vertex
    attributes::Vector{VertexAttribute}
end

get_default_attributes() = [
    VertexAttribute(0, 2, GL_FLOAT, false, 0),
    VertexAttribute(1, 2, GL_FLOAT, false, 2 * sizeof(Float32))
]

get_default_3d_attributes() = [
    VertexAttribute(0, 3, GL_FLOAT, false, 0),                   # Position (x, y, z)
    VertexAttribute(2, 2, GL_FLOAT, false, 6 * sizeof(Float32)), # Texcoord (u, v)
    VertexAttribute(1, 3, GL_FLOAT, false, 3 * sizeof(Float32))  # Normal (nx, ny, nz)
]

function create_mesh(vertices::Vector{T},
                     attributes::Vector{VertexAttribute} = get_default_attributes();
                     draw_mode::GLenum = GL_TRIANGLES) where T

    # Create and bind VAO
    vao = gl_gen_vertex_array()
    glBindVertexArray(vao)

    # Create VBO for vertex data
    vbo = gl_gen_buffer()
    glBindBuffer(GL_ARRAY_BUFFER, vbo)

    # Upload vertex data
    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(vertices),
        vertices,
        GL_STATIC_DRAW
    )

    # Calculate stride (size of vertex in bytes)
    stride = sum([attr.size * sizeof(T) for attr in attributes])

    # Set up vertex attributes
    for attr in attributes
        glEnableVertexAttribArray(attr.location)
        glVertexAttribPointer(
            attr.location,
            attr.size,
            attr.type,
            attr.normalized ? GL_TRUE : GL_FALSE,
            stride,
            Ptr{Cvoid}(attr.offset)
        )
    end

    # Unbind VAO to prevent accidental modification
    glBindVertexArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # Create and return mesh
    return Mesh(
        vao,
        vbo,
        length(vertices),
        draw_mode,
        stride,
        attributes
    )
end

#=
function draw_mesh(mesh::Mesh, shader_program::GLuint)
    glUseProgram(shader_program)
    glBindVertexArray(mesh.vao)
    glDrawArrays(mesh.draw_mode, 0, mesh.vertex_count)
    glBindVertexArray(0)
    glUseProgram(0)
end
draw_mesh(mesh::Mesh, shader_info::ShaderInfo) = draw_mesh(mesh, shader_info.program_id)
=#

function draw_mesh(mesh::Mesh, texture_id::GLuint, tint_color::Vector{Float32}=[1.0f0, 1.0f0, 1.0f0])
    ctx::RenderContext = get_context()
    glUseProgram(ctx.texture_shader.program_id)
    glUniformMatrix4fv(ctx.texture_shader.uniform_locations["projection"], 1, GL_FALSE, get_context().projection)
    transform::Matrix = get_context().context_stack[end].transform
    glUniformMatrix4fv(ctx.texture_shader.uniform_locations["model"], 1, GL_FALSE, transform)
    glUniform3f(ctx.texture_shader.uniform_locations["tintColor"], tint_color[1], tint_color[2], tint_color[3])

    # Activate texture unit 0 and bind the texture
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glUniform1i(ctx.texture_shader.uniform_locations["textureSampler"], 0)

    # Bind the mesh's VAO (which contains all vertex attribute configurations)
    glBindVertexArray(mesh.vao)

    # Draw the mesh
    glDrawArrays(mesh.draw_mode, 0, mesh.vertex_count)

    # Clean up state
    glBindTexture(GL_TEXTURE_2D, 0)
    glBindVertexArray(0)
    glUseProgram(0)
end

function draw_mesh(mesh::Mesh, tint_color::Vector{Float32}=[1.0f0, 1.0f0, 1.0f0])
    ctx::RenderContext = get_context()
    draw_mesh(mesh, ctx.blank_texture, tint_color)
end

# Update vertex data (for dynamic meshes)
function update_mesh_vertices!(mesh::Mesh, vertices::Vector{T}, usage::GLenum = GL_DYNAMIC_DRAW) where T
    glBindVertexArray(mesh.vao)
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo)
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, usage)
    glBindVertexArray(0)
    mesh.vertex_count = length(vertices)
end

# Destroy mesh and free GPU resources
function destroy_mesh(mesh::Mesh)
    # Delete VBO
    if mesh.vbo != 0
        glDeleteBuffers(1, [mesh.vbo])
    end

    # Delete VAO
    if mesh.vao != 0
        glDeleteVertexArrays(1, [mesh.vao])
    end

    # Clear fields to prevent accidental use after destruction
    mesh.vao = 0
    mesh.vbo = 0
    mesh.vertex_count = 0
end

# Example: Create a quad mesh using triangles
function create_quad(width::Float32, height::Float32)
    # Positions (x, y) and texture coordinates (u, v) for 6 vertices (2 triangles)
    x = width / 2
    y = height / 2
    vertices = Float32[
        -x, -y, 0, 0,
        x, -y, 1, 0,
        -x, y, 0, 1,
        x, -y, 1, 0,
        -x, y, 0, 1,
        x, y, 1, 1
    ]

    return create_mesh(vertices)
end

function create_circle(radius::Float32, segments::Int = 32)
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

    return create_mesh(vertices)
end
