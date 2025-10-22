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
    VertexAttribute(0, 3, GL_FLOAT, false, 0),                    # Position (x, y, z)
    VertexAttribute(1, 2, GL_FLOAT, false, 3 * sizeof(Float32))   # Texture coordinates (u, v)]
]

function create_mesh(vertices::Vector{T} = Float32[0, 0, 0, 0],
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
        count_vertices(attributes, vertices),
        draw_mode,
        stride,
        attributes
    )
end

function create_3d_mesh(vertices::Vector{Float32} = Float32[0, 0, 0, 0, 0]; kwargs...)
    return create_mesh(vertices, get_default_3d_attributes(); kwargs...)
end

function draw_mesh(mesh::Mesh, texture_id::GLuint, tint_color::Vector{Float32})
    ctx::RenderContext = get_context()
    glUseProgram(ctx.shader.program_id)
    glUniformMatrix4fv(ctx.shader.uniform_locations["projection"], 1, GL_FALSE, get_state().projection)
    glUniformMatrix4fv(ctx.shader.uniform_locations["view"], 1, GL_FALSE, get_state().view)
    glUniformMatrix4fv(ctx.shader.uniform_locations["model"], 1, GL_FALSE, get_state().transform)
    glUniform4f(ctx.shader.uniform_locations["color"], tint_color...)

    # Activate texture unit 0 and bind the texture
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glUniform1i(ctx.shader.uniform_locations["textureSampler"], 0)

    # Bind the mesh's VAO (which contains all vertex attribute configurations)
    glBindVertexArray(mesh.vao)

    # Draw the mesh
    glDrawArrays(mesh.draw_mode, 0, mesh.vertex_count)

    # Clean up state
    glBindTexture(GL_TEXTURE_2D, 0)
    glBindVertexArray(0)
    glUseProgram(0)
end

function draw_mesh(mesh::Mesh, texture_id::GLuint)
    draw_mesh(mesh, texture_id, Float32[get_state().fill_color...])
end

function draw_mesh(mesh::Mesh, tint_color::Vector{Float32})
    ctx::RenderContext = get_context()
    draw_mesh(mesh, ctx.blank_texture, tint_color)
end

function draw_mesh(mesh::Mesh)
    ctx::RenderContext = get_context()
    draw_mesh(mesh, ctx.blank_texture)
end

function count_vertices(attributes::Vector, vertices::Vector)
    stride::Int = sum([attr.size for attr in attributes])
    return div(length(vertices), stride)
end

# Update vertex data (for dynamic meshes)
function update_mesh_vertices!(mesh::Mesh, vertices::Vector{Float32}, usage::GLenum = GL_DYNAMIC_DRAW)
    glBindVertexArray(mesh.vao)
    glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo)
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, usage)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    mesh.vertex_count = count_vertices(mesh.attributes, vertices)
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

    return create_mesh(vertices)
end

function create_cube(size::Number = 1.0)
    # Half size for centering
    s::Float32 = Float32(size / 2)

    # Vertices for all 6 faces of the cube (12 triangles total)
    # Each vertex has: x, y, z, u, v (position + texture coordinates)
    vertices = Float32[
        # Front face (2 triangles)
        -s, -s,  s, 0, 0,  # Bottom-left
         s, -s,  s, 1, 0,  # Bottom-right
        -s,  s,  s, 0, 1,  # Top-left
         s, -s,  s, 1, 0,  # Bottom-right
        -s,  s,  s, 0, 1,  # Top-left
         s,  s,  s, 1, 1,  # Top-right

        # Back face (2 triangles)
         s, -s, -s, 0, 0,  # Bottom-left
        -s, -s, -s, 1, 0,  # Bottom-right
         s,  s, -s, 0, 1,  # Top-left
        -s, -s, -s, 1, 0,  # Bottom-right
         s,  s, -s, 0, 1,  # Top-left
        -s,  s, -s, 1, 1,  # Top-right

        # Left face (2 triangles)
        -s, -s, -s, 0, 0,  # Bottom-left
        -s, -s,  s, 1, 0,  # Bottom-right
        -s,  s, -s, 0, 1,  # Top-left
        -s, -s,  s, 1, 0,  # Bottom-right
        -s,  s, -s, 0, 1,  # Top-left
        -s,  s,  s, 1, 1,  # Top-right

        # Right face (2 triangles)
         s, -s,  s, 0, 0,  # Bottom-left
         s, -s, -s, 1, 0,  # Bottom-right
         s,  s,  s, 0, 1,  # Top-left
         s, -s, -s, 1, 0,  # Bottom-right
         s,  s,  s, 0, 1,  # Top-left
         s,  s, -s, 1, 1,  # Top-right

        # Top face (2 triangles)
        -s,  s,  s, 0, 0,  # Bottom-left
         s,  s,  s, 1, 0,  # Bottom-right
        -s,  s, -s, 0, 1,  # Top-left
         s,  s,  s, 1, 0,  # Bottom-right
        -s,  s, -s, 0, 1,  # Top-left
         s,  s, -s, 1, 1,  # Top-right

        # Bottom face (2 triangles)
        -s, -s, -s, 0, 0,  # Bottom-left
         s, -s, -s, 1, 0,  # Bottom-right
        -s, -s,  s, 0, 1,  # Top-left
         s, -s, -s, 1, 0,  # Bottom-right
        -s, -s,  s, 0, 1,  # Top-left
         s, -s,  s, 1, 1   # Top-right
    ]

    return create_3d_mesh(vertices)
end

function create_uv_sphere(radius::Number = 1.0, u_segments::Int = 32, v_segments::Int = 16)
    vertices = Float32[]
    r::Float32 = Float32(radius)
    
    for i in 0:v_segments-1
        v1 = Float32(i) / Float32(v_segments) * pi
        v2 = Float32(i + 1) / Float32(v_segments) * pi
        
        for j in 0:u_segments-1
            u1 = Float32(j) / Float32(u_segments) * 2pi
            u2 = Float32(j + 1) / Float32(u_segments) * 2pi
            
            # Calculate the four corners of the current quad
            # Bottom-left
            x1 = r * cos(u1) * sin(v1)
            y1 = r * cos(v1) 
            z1 = r * sin(u1) * sin(v1)
            
            # Bottom-right  
            x2 = r * cos(u2) * sin(v1)
            y2 = r * cos(v1)
            z2 = r * sin(u2) * sin(v1)
            
            # Top-left
            x3 = r * cos(u1) * sin(v2)
            y3 = r * cos(v2)
            z3 = r * sin(u1) * sin(v2)
            
            # Top-right
            x4 = r * cos(u2) * sin(v2)
            y4 = r * cos(v2)
            z4 = r * sin(u2) * sin(v2)
            
            # UV coordinates
            tex_u1 = Float32(j) / Float32(u_segments)
            tex_u2 = Float32(j + 1) / Float32(u_segments)
            tex_v1 = Float32(i) / Float32(v_segments)
            tex_v2 = Float32(i + 1) / Float32(v_segments)
            
            # Avoid degenerate triangles at poles
            if i > 0  # Not at north pole
                # First triangle: bottom-left, bottom-right, top-left
                append!(vertices, [x1, y1, z1, tex_u1, tex_v1])
                append!(vertices, [x2, y2, z2, tex_u2, tex_v1])
                append!(vertices, [x3, y3, z3, tex_u1, tex_v2])
            end
            
            if i < v_segments - 1  # Not at south pole
                # Second triangle: bottom-right, top-right, top-left
                append!(vertices, [x2, y2, z2, tex_u2, tex_v1])
                append!(vertices, [x4, y4, z4, tex_u2, tex_v2])
                append!(vertices, [x3, y3, z3, tex_u1, tex_v2])
            end
        end
    end
    
    return create_3d_mesh(vertices)
end

function load_obj_mesh(filepath::String)
    vertices = Float32[]
    positions = Vector{Vector{Float32}}()
    texcoords = Vector{Vector{Float32}}()

    open(filepath, "r") do f
        for line in eachline(f)
            parts = split(line)
            if isempty(parts) continue end

            if parts[1] == "v"
                push!(positions, [parse(Float32, parts[2]), parse(Float32, parts[3]), parse(Float32, parts[4])])
            elseif parts[1] == "vt"
                push!(texcoords, [parse(Float32, parts[2]), parse(Float32, parts[3])])
            elseif parts[1] == "f"
                # Triangulate faces with more than 3 vertices (N-gons)
                # Assumes 1-based indexing for OBJ.
                face_vertices = []
                for i in 2:length(parts)
                    v_vt_n = split(parts[i], '/')
                    v_idx = parse(Int, v_vt_n[1])
                    vt_idx = parse(Int, v_vt_n[2])
                    push!(face_vertices, (v_idx, vt_idx))
                end

                # Fan triangulation
                if length(face_vertices) >= 3
                    # First vertex of the face is the common vertex for the fan
                    v1_idx, vt1_idx = face_vertices[1]
                    pos1 = positions[v1_idx]
                    tc1 = texcoords[vt1_idx]

                    for i in 2:(length(face_vertices) - 1)
                        # Second vertex of the triangle
                        v2_idx, vt2_idx = face_vertices[i]
                        pos2 = positions[v2_idx]
                        tc2 = texcoords[vt2_idx]

                        # Third vertex of the triangle
                        v3_idx, vt3_idx = face_vertices[i+1]
                        pos3 = positions[v3_idx]
                        tc3 = texcoords[vt3_idx]

                        append!(vertices, pos1[1], pos1[2], pos1[3], tc1[1], tc1[2])
                        append!(vertices, pos2[1], pos2[2], pos2[3], tc2[1], tc2[2])
                        append!(vertices, pos3[1], pos3[2], pos3[3], tc3[1], tc3[2])
                    end
                end
            end
        end
    end
    
    return create_3d_mesh(vertices)
end
