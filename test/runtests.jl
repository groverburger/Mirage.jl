using Test

include("MirageTestDemos.jl")
using .MirageTestDemos

function _env_flag(name::String, default::Bool)
    value = lowercase(strip(get(ENV, name, default ? "1" : "0")))
    return value in ("1", "true", "yes", "on")
end

function _interactive_tests_enabled()
    default = !_env_flag("CI", false)
    return _env_flag("MIRAGE_TEST_INTERACTIVE", default)
end

function _selected_demos()
    value = strip(get(ENV, "MIRAGE_TEST_DEMOS", ""))
    isempty(value) && return Symbol[]
    return Symbol.(strip.(split(value, ",")))
end

@testset "Mirage" begin
    api_behavior_tests()

    if _interactive_tests_enabled()
        @info "Running interactive Mirage demo windows. Close each window to advance." demos = getfield.(demos(), :name)
        run_all_demos(_selected_demos())
    else
        @info "Skipping interactive Mirage demo windows. Set MIRAGE_TEST_INTERACTIVE=1 to run them."
        @test true
    end
end
