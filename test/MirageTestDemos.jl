module MirageTestDemos

using Test
using LinearAlgebra
import Mirage

export Demo, demos, run_all_demos, spinning_square, two_d_primitives, basic_3d_scene,
       lit_textured_3d_scene, canvas_and_texture, api_behavior_tests

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

struct Demo
    name::Symbol
    title::String
    run::Function
end

function _resource_path(parts...)
    return joinpath(REPO_ROOT, parts...)
end

function _initialize_demo(title::String; width::Int = 800, height::Int = 600)
    @info "Opening Mirage demo window. Close it to continue." title
    return Mirage.initialize(
        window_width = width,
        window_height = height,
        window_title = title,
    )
end

function _render_loop(render::Function; wait_for_events::Bool = false)
    frame_count = 0
    while !Mirage.GLFW.WindowShouldClose(Mirage.window[])
        frame_count += 1
        Mirage.clear()
        render()
        Mirage.GLFW.SwapBuffers(Mirage.window[])
        wait_for_events ? Mirage.GLFW.WaitEvents() : Mirage.GLFW.PollEvents()
        Mirage.gl_check_error("end of demo frame $frame_count")
    end
    return nothing
end

function _terminate_demo_window()
    if isdefined(Mirage, :terminate)
        Mirage.terminate()
    end
    return nothing
end

function _window_size()
    ctx = Mirage.get_context()
    return Float32(ctx.width / ctx.dpi_scaling), Float32(ctx.height / ctx.dpi_scaling)
end

function _label(message::String, x::Number, y::Number; color = Mirage.rgba(230, 235, 245))
    Mirage.save()
    Mirage.translate(x, y)
    Mirage.fillcolor(color)
    Mirage.text(message)
    Mirage.restore()
end

function _checker_texture()
    data = Float32[
        0.10, 0.12, 0.18, 1.0,  0.95, 0.95, 0.80, 1.0,
        0.95, 0.95, 0.80, 1.0,  0.15, 0.36, 0.55, 1.0,
    ]
    texture_id = Mirage.gl_gen_texture()
    Mirage.glBindTexture(Mirage.GL_TEXTURE_2D, texture_id)
    Mirage.glTexParameteri(Mirage.GL_TEXTURE_2D, Mirage.GL_TEXTURE_WRAP_S, Mirage.GL_REPEAT)
    Mirage.glTexParameteri(Mirage.GL_TEXTURE_2D, Mirage.GL_TEXTURE_WRAP_T, Mirage.GL_REPEAT)
    Mirage.glTexParameteri(Mirage.GL_TEXTURE_2D, Mirage.GL_TEXTURE_MIN_FILTER, Mirage.GL_NEAREST)
    Mirage.glTexParameteri(Mirage.GL_TEXTURE_2D, Mirage.GL_TEXTURE_MAG_FILTER, Mirage.GL_NEAREST)
    Mirage.glTexImage2D(
        Mirage.GL_TEXTURE_2D,
        0,
        Mirage.GL_RGBA32F,
        2,
        2,
        0,
        Mirage.GL_RGBA,
        Mirage.GL_FLOAT,
        data,
    )
    Mirage.glBindTexture(Mirage.GL_TEXTURE_2D, 0)
    return texture_id
end

function api_behavior_tests()
    @testset "Mirage CPU-side API" begin
        m = Matrix{Float32}(I, 4, 4)
        Mirage.translate!(m, 2, 3, 4)
        @test m[:, 4] == Float32[2, 3, 4, 1]

        m = Matrix{Float32}(I, 4, 4)
        Mirage.scale!(m, 2, 3, 4)
        @test m[1, 1] == 2
        @test m[2, 2] == 3
        @test m[3, 3] == 4

        m = Matrix{Float32}(I, 4, 4)
        Mirage.rotate!(m, Float32(pi / 2))
        @test m[1, 1] ≈ 0 atol = 1f-6
        @test m[2, 1] ≈ 1 atol = 1f-6

        @test Mirage.rgba(255, 128, 0, 64) ==
              (1.0f0, Float32(128 / 255), 0.0f0, Float32(64 / 255))

        p = Mirage.perspective(Float32(pi / 4), 4.0f0 / 3.0f0, 0.01f0, 100.0f0)
        @test size(p) == (4, 4)
        @test p[4, 3] == -1
    end
end

function spinning_square()
    _initialize_demo("Mirage Test 1 - README Spinning Square")

    Mirage.start_render_loop(function ()
        width, height = _window_size()

        Mirage.save()
        Mirage.translate(width / 2, height / 2)
        Mirage.rotate(time())
        Mirage.fillcolor(Mirage.rgba(255, 40, 40))
        Mirage.rect(-55, -55, 110, 110)
        Mirage.fill()
        Mirage.restore()

        _label("README spinning square", 20, 24)
        _label("Close this window for the next demo", 20, height - 36;
               color = Mirage.rgba(180, 205, 255))
    end)

    return nothing
end

function two_d_primitives()
    _initialize_demo("Mirage Test 2 - 2D Primitives, Paths, Text")

    start_time = time()
    Mirage.start_render_loop(function ()
        width, height = _window_size()
        t = time() - start_time

        Mirage.save()
        Mirage.fillcolor(Mirage.rgba(28, 36, 46))
        Mirage.fillrect(0, 0, width, height)
        Mirage.restore()

        Mirage.save()
        Mirage.translate(width * 0.25, height * 0.32)
        Mirage.rotate(t)
        Mirage.fillcolor(Mirage.rgba(255, 190, 80))
        Mirage.rect(-70, -45, 140, 90)
        Mirage.fill()
        Mirage.restore()

        Mirage.save()
        Mirage.translate(width * 0.62, height * 0.34)
        Mirage.scale(1.0 + 0.18 * sin(2t))
        Mirage.fillcolor(Mirage.rgba(85, 205, 150, 190))
        Mirage.fillcircle(76)
        Mirage.restore()

        Mirage.save()
        Mirage.beginpath()
        Mirage.moveto(80, height * 0.68)
        Mirage.lineto(width * 0.35, height * 0.58)
        Mirage.lineto(width * 0.52, height * 0.74)
        Mirage.lineto(width * 0.78, height * 0.60)
        Mirage.strokecolor(Mirage.rgba(105, 165, 255))
        Mirage.strokewidth(12)
        Mirage.stroke()
        Mirage.restore()

        Mirage.save()
        Mirage.translate(width * 0.78, height * 0.32)
        Mirage.rotate(-0.35 + 0.12sin(t))
        Mirage.beginpath()
        Mirage.moveto(-58, -42)
        Mirage.lineto(58, -42)
        Mirage.lineto(76, 8)
        Mirage.lineto(18, 64)
        Mirage.lineto(-68, 34)
        Mirage.closepath()
        Mirage.fillcolor(Mirage.rgba(135, 85, 215, 120))
        Mirage.strokecolor(Mirage.rgba(245, 235, 255))
        Mirage.strokewidth(7)
        Mirage.fill()
        Mirage.stroke()
        Mirage.restore()

        Mirage.save()
        Mirage.translate(26, 30)
        Mirage.scale(2)
        Mirage.fillcolor(Mirage.rgba(245, 245, 245))
        Mirage.text("2D primitives, paths, text")
        Mirage.restore()

        _label("fillrect / rect+fill / fillcircle / stroke / text", 28, height - 38;
               color = Mirage.rgba(200, 220, 255))
    end)

    return nothing
end

function canvas_and_texture()
    _initialize_demo("Mirage Test 3 - Textures and Canvas")

    canvas = Mirage.create_canvas(192, 192)
    texture_id = UInt32(0)
    test_image = _resource_path("testimage.jpg")

    try
        texture_id = isfile(test_image) ? Mirage.load_texture(test_image) : _checker_texture()

        Mirage.set_canvas(canvas)
        Mirage.clear()
        Mirage.update_ortho_projection_matrix(canvas.width, canvas.height, 1.0)
        Mirage.fillcolor(Mirage.rgba(20, 28, 42))
        Mirage.fillrect(0, 0, canvas.width, canvas.height)
        Mirage.save()
        Mirage.translate(24, 72)
        Mirage.scale(3)
        Mirage.fillcolor(Mirage.rgba(255, 255, 255))
        Mirage.text("Canvas")
        Mirage.restore()
        Mirage.set_canvas()

        start_time = time()
        _render_loop(function ()
            width, height = _window_size()
            t = time() - start_time
            size = min(width, height) * 0.34

            Mirage.save()
            Mirage.translate(width * 0.24, height * 0.28)
            Mirage.scale(1.0 + 0.08sin(2t))
            Mirage.fillcolor(Mirage.rgba(255, 255, 255))
            Mirage.drawimage(-size / 2, -size / 2, size, size, texture_id)
            Mirage.restore()

            Mirage.save()
            Mirage.translate(width * 0.68, height * 0.52)
            Mirage.rotate(t * 0.7)
            Mirage.fillcolor(Mirage.rgba(255, 255, 255))
            Mirage.drawimage(-size / 2, -size / 2, size, size, canvas.texture)
            Mirage.restore()

            _label("Loaded image or generated checker texture", 26, 28)
            _label("Offscreen canvas rendered back into the window", 26, height - 36;
                   color = Mirage.rgba(200, 220, 255))
        end)
    finally
        if texture_id != 0
            Mirage.destroy_texture!(texture_id)
        end
        Mirage.destroy!(canvas)
        _terminate_demo_window()
    end

    return nothing
end

function basic_3d_scene()
    _initialize_demo("Mirage Test 4 - Basic 3D Scene")

    cube = Mirage.create_cube(2.0)
    sphere = Mirage.create_uv_sphere(0.9, 24, 12)

    try
        Mirage.glEnable(Mirage.GL_DEPTH_TEST)
        start_time = time()

        _render_loop(function ()
            width, height = _window_size()
            t = time() - start_time

            Mirage.glEnable(Mirage.GL_DEPTH_TEST)
            Mirage.update_perspective_projection_matrix(; near = 0.01, far = 100.0, fov = pi / 4)
            camera = Float32[6cos(t * 0.35), 6sin(t * 0.35), 3.0]
            Mirage.lookat(camera, Float32[0, 0, 0], Float32[0, 0, 1])

            Mirage.save()
            Mirage.translate(-1.6, 0, 0)
            Mirage.rotate(t * 0.8, t * 0.45, t * 0.25)
            Mirage.fillcolor(Mirage.rgba(255, 125, 80))
            Mirage.draw_mesh(cube)
            Mirage.restore()

            Mirage.save()
            Mirage.translate(1.7, 0, 0)
            Mirage.rotate(0, t * 0.7, t)
            Mirage.fillcolor(Mirage.rgba(80, 175, 255))
            Mirage.draw_mesh(sphere)
            Mirage.restore()

            Mirage.glDisable(Mirage.GL_DEPTH_TEST)
            Mirage.update_ortho_projection_matrix()
            _label("Perspective camera, cube mesh, UV sphere mesh", 22, 28)
            _label("Close this window to finish Pkg.test()", 22, height - 36;
                   color = Mirage.rgba(200, 220, 255))
        end)
    finally
        Mirage.glDisable(Mirage.GL_DEPTH_TEST)
        Mirage.destroy!(sphere)
        Mirage.destroy!(cube)
        _terminate_demo_window()
    end

    return nothing
end

function lit_textured_3d_scene()
    _initialize_demo("Mirage Test 5 - Lit Textured 3D and In-Scene 2D")

    cube = Mirage.create_cube(2.0)
    sphere = Mirage.create_uv_sphere(0.85, 32, 16)
    texture_id = UInt32(0)

    vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec2 aTexCoord;
        layout (location = 2) in vec3 aNormal;

        out vec3 FragPos;
        out vec2 TexCoord;
        out vec3 Normal;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        void main()
        {
            vec4 worldPos = model * vec4(aPos, 1.0);
            FragPos = worldPos.xyz;
            TexCoord = aTexCoord;
            Normal = mat3(transpose(inverse(model))) * aNormal;
            gl_Position = projection * view * worldPos;
        }
    """

    fragment_shader = """
        #version 330 core
        in vec3 FragPos;
        in vec2 TexCoord;
        in vec3 Normal;

        out vec4 FragColor;

        uniform sampler2D textureSampler;
        uniform vec3 lightPos;
        uniform vec3 viewPos;
        uniform vec3 lightColor;
        uniform vec4 tintColor;

        void main()
        {
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPos - FragPos);
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);

            float diffuseStrength = max(dot(norm, lightDir), 0.0);
            float specularStrength = pow(max(dot(viewDir, reflectDir), 0.0), 32.0) * 0.45;

            vec3 lighting = (0.18 + diffuseStrength) * lightColor + specularStrength * lightColor;
            vec4 texel = texture(textureSampler, TexCoord) * tintColor;
            FragColor = vec4(texel.rgb * lighting, texel.a);
        }
    """

    shader = Mirage.create_shader_program(vertex_shader, fragment_shader)
    for uniform in ("model", "view", "projection", "textureSampler", "lightPos",
                    "viewPos", "lightColor", "tintColor")
        Mirage.initialize_shader_uniform!(shader, uniform)
    end

    try
        texture_path = _resource_path("testimage.jpg")
        texture_id = isfile(texture_path) ? Mirage.load_texture(texture_path) : _checker_texture()
        Mirage.glEnable(Mirage.GL_DEPTH_TEST)

        start_time = time()
        _render_loop(function ()
            width, height = _window_size()
            t = time() - start_time
            camera = Float32[5cos(t * 0.25), 6sin(t * 0.25), 3.2]
            light = Float32[3.5cos(t), 3.5sin(t), 4.0]

            Mirage.glEnable(Mirage.GL_DEPTH_TEST)
            Mirage.update_perspective_projection_matrix(; near = 0.01, far = 100.0, fov = pi / 4)
            Mirage.lookat(camera, Float32[0, 0, 0], Float32[0, 0, 1])

            Mirage.save()
            Mirage.translate(-1.35, 0, 0)
            Mirage.rotate(t * 0.45, t * 0.55, t * 0.2)
            Mirage.draw_mesh(cube, shader, s -> begin
                Mirage.set_uniform(s, "model", Mirage.get_state().transform)
                Mirage.set_uniform(s, "view", Mirage.get_state().view)
                Mirage.set_uniform(s, "projection", Mirage.get_state().projection)
                Mirage.set_uniform(s, "lightPos", light)
                Mirage.set_uniform(s, "viewPos", camera)
                Mirage.set_uniform(s, "lightColor", Float32[1.0, 0.96, 0.86])
                Mirage.set_uniform(s, "tintColor", Float32[1.0, 0.82, 0.72, 1.0])
                Mirage.glActiveTexture(Mirage.GL_TEXTURE0)
                Mirage.glBindTexture(Mirage.GL_TEXTURE_2D, texture_id)
                Mirage.set_uniform(s, "textureSampler", 0)
            end)
            Mirage.restore()

            Mirage.save()
            Mirage.translate(1.85, -1.15, 0.35)
            Mirage.rotate(t * 0.2, t * 0.8, t * 0.35)
            Mirage.draw_mesh(sphere, shader, s -> begin
                Mirage.set_uniform(s, "model", Mirage.get_state().transform)
                Mirage.set_uniform(s, "view", Mirage.get_state().view)
                Mirage.set_uniform(s, "projection", Mirage.get_state().projection)
                Mirage.set_uniform(s, "lightPos", light)
                Mirage.set_uniform(s, "viewPos", camera)
                Mirage.set_uniform(s, "lightColor", Float32[1.0, 0.98, 0.9])
                Mirage.set_uniform(s, "tintColor", Float32[0.72, 1.0, 0.78, 1.0])
                Mirage.glActiveTexture(Mirage.GL_TEXTURE0)
                Mirage.glBindTexture(Mirage.GL_TEXTURE_2D, texture_id)
                Mirage.set_uniform(s, "textureSampler", 0)
            end)
            Mirage.restore()

            Mirage.glBindTexture(Mirage.GL_TEXTURE_2D, 0)

            Mirage.save()
            Mirage.translate(-2.65, -2.1, 1.32)
            Mirage.rotate(pi / 2, 0, 0)
            Mirage.scale(0.018)
            Mirage.beginpath()
            Mirage.moveto(0, 0)
            Mirage.lineto(88, -36)
            Mirage.lineto(168, -4)
            Mirage.lineto(238, -44)
            Mirage.strokecolor(Mirage.rgba(255, 210, 95, 255))
            Mirage.strokewidth(5)
            Mirage.stroke()
            Mirage.restore()

            Mirage.save()
            Mirage.translate(-2.7, -2.12, 1.7)
            Mirage.rotate(pi / 2, 0, pi)
            Mirage.scale(0.016)
            Mirage.fillcolor(Mirage.rgba(245, 248, 255))
            Mirage.text("Phong lighting + texture")
            Mirage.restore()
        end)
    finally
        Mirage.glDisable(Mirage.GL_DEPTH_TEST)
        if texture_id != 0
            Mirage.destroy_texture!(texture_id)
        end
        Mirage.glDeleteProgram(shader.program_id)
        Mirage.destroy!(sphere)
        Mirage.destroy!(cube)
        _terminate_demo_window()
    end

    return nothing
end

function demos()
    return Demo[
        Demo(:spinning_square, "README spinning square", spinning_square),
        Demo(:two_d_primitives, "2D primitives, paths, and text", two_d_primitives),
        Demo(:canvas_and_texture, "Textures and offscreen canvas", canvas_and_texture),
        Demo(:basic_3d_scene, "Basic 3D scene", basic_3d_scene),
        Demo(:lit_textured_3d_scene, "Lit textured 3D scene with in-scene 2D", lit_textured_3d_scene),
    ]
end

function run_all_demos(selected::Vector{Symbol} = Symbol[])
    wanted = isempty(selected) ? demos() : filter(demo -> demo.name in selected, demos())
    missing = setdiff(selected, getfield.(demos(), :name))
    isempty(missing) || error("Unknown Mirage demo(s): $(join(string.(missing), ", "))")

    for demo in wanted
        @testset "$(demo.title)" begin
            @test isnothing(demo.run())
        end
    end

    return nothing
end

end # module MirageTestDemos
