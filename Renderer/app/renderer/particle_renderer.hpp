#pragma once
#include "./window_wrapper.hpp"
#include "./transforms/transform.hpp"
#include "glcore/vertexarray.hpp"
#include "glcore/shader.h"

class ParticleRenderer
{
    public:
    struct Vert_Arr
    {
        std::unique_ptr<
        nitros::glcore::VertexArray>    vert_array;
        bool is_point = true;
    };

    ParticleRenderer(Window  &window_);
    ~ParticleRenderer() = default;

    auto insert_vertex_meshes(Vert_Arr  &&vao) -> std::size_t;
    auto get_vert_arr(std::uint32_t  index_) -> Vert_Arr* {
        return &_vert_arrays.at(index_);
    }
    void render();
    auto is_key_pressed(int  glfw_key) -> bool;
    auto shader() -> nitros::glcore::Shader& {
        return *_shader;
    }

    private:
    Window&                     _window;
    std::vector<Vert_Arr>       _vert_arrays;
    nitros::SRT                 _srt;
    std::unique_ptr<
    nitros::glcore::Shader>     _shader;
    std::unique_ptr<
    nitros::glcore::Shader>     _fade_shader;
    std::uint32_t               _dummy_vao = 0;
};