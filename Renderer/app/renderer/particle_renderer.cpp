#include <glad/glad.h>
#include "./particle_renderer.hpp"
#include "./shader_header.hpp"
#include "glcore/framebuffer.hpp"
#include "glcore/rasterizer.hpp"
#include "glcore/context.hpp"
#include <iostream>

auto get_shader_code() -> std::pair<std::string, std::string>
{
    const auto vertex_shader1 =  R"( layout (location = 0)  in vec3 position;
                                layout (location = 1)  in vec3 color;
                                uniform mat4 mvp;
                                uniform float pointSize;
                                uniform float isPoint;
                                out vec3 VColor;
                                out float fIsPoint;
                                void main(void)
                                {                
                                    gl_Position = mvp * vec4(position, 1.0);  
                                    VColor = color;
                                    fIsPoint = isPoint;
                                    if (isPoint > 0.5) {
                                        gl_PointSize = pointSize / gl_Position.w;
                                    } else {
                                        gl_PointSize = 1.0;
                                    }
                                } )";

    const auto frag_shader1 =   R"( in vec3 VColor;
                                in float fIsPoint;
                                out vec4 color;     
                                void main(void) 
                                {  
                                    if (fIsPoint > 0.5) {
                                        vec2 coord = gl_PointCoord * 2.0 - 1.0;
                                        float dist = length(coord);
                                        if (dist > 1.0) {
                                            discard;
                                        }
                                        float alpha = 1.0 - smoothstep(0.9, 1.0, dist);
                                        float glow = 1.0 - dist;
                                        color = vec4(VColor * (glow * 0.5 + 0.5), alpha * glow);
                                    } else {
                                        color = vec4(VColor, 1.0);
                                    }
                                } )";
                                
    return {vertex_shader1, frag_shader1};
}    

ParticleRenderer::ParticleRenderer(Window  &window_)
    :_window{window_}
    ,_vert_arrays{[]() -> std::vector<Vert_Arr>
    {
        using namespace nitros;
        #if !defined(__EMSCRIPTEN__)
            glcore::load_context(glfwGetProcAddress);
        #endif
        return {};
    }()}
    ,_srt{ glm::vec3{1,1,1}, glm::vec3{0, 0, 0}, glm::vec3{0, 0, 0} }
    ,_shader{[]()
    {
        auto [vertex_shader1, frag_shader1] = get_shader_code();
        auto shader = std::make_unique<nitros::glcore::Shader>(vert_header + vertex_shader1, frag_header + frag_shader1);
        return shader;
    }()}
{
    using namespace nitros;

    auto&& raster = glcore::Rasterizer::get_instance();
    raster.set_cull_face(glcore::Rasterizer::cull::front_and_back);
    raster.set_polygonMode(glcore::Rasterizer::polygonMode::fill);
    raster.set_point_size(5);

    std::cout<< glcore::get_renderer() << std::endl;;
    _window.camera.Position = {0.0f, 2.0f, 6.0f};
    _window.camera.ProcessLook(Camera_Look::Down, 0.4f); // Tilt camera down towards the box

    // Enable blending and program point size
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // Create dummy VAO for fullscreen triangle
    glGenVertexArrays(1, &_dummy_vao);

    // Compile fade shader for motion blur
    std::string fade_vert = vert_header + R"(
        void main() {
            float x = -1.0 + float((gl_VertexID & 1) << 2);
            float y = -1.0 + float((gl_VertexID & 2) << 1);
            gl_Position = vec4(x, y, 0.0, 1.0);
        }
    )";
    std::string fade_frag = frag_header + R"(
        out vec4 fragColor;
        void main() {
            fragColor = vec4(0.1, 0.1, 0.1, 0.7); // higher alpha for shorter trails (cleaner view)
        }
    )";
    _fade_shader = std::make_unique<nitros::glcore::Shader>(fade_vert, fade_frag);
}

auto ParticleRenderer::insert_vertex_meshes(Vert_Arr  &&vao) -> std::size_t
{
    _vert_arrays.push_back(std::move(vao));
    return _vert_arrays.size() - 1;
}

void ParticleRenderer::render()
{
    using namespace nitros;

    auto&& frame_buffer = glcore::FrameBuffer::get_default();
    frame_buffer.bind();
    
    // Clear only depth buffer to keep previous color frames for motion blur
    frame_buffer.clear_depth();

    // Draw fade quad to fade the color buffer
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    _fade_shader->use();
    glBindVertexArray(_dummy_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEnable(GL_DEPTH_TEST);

    auto delta_time = Time2::delta_time();
    float dt = delta_time.count() / 1000.0f;

    auto&& srt = _srt;

    srt.rotate.y += (glm::radians(10.f)*dt);
    auto model = model_matrix(srt);

    _shader->use();
    auto&& cam    = _window.camera;
    auto view_mat = cam.GetViewMatrix();

    auto [width, height] = _window.window_dim();

    auto _fovy      = 45.0f;
    auto near_plane = 0.1f;
    auto far_plane  = 200.0f;
    auto projection = perspective_projection_matrix(_fovy, static_cast<float>(width), static_cast<float>(height), near_plane, far_plane);

    _shader->set_uniform_matrix4fv("mvp", projection * view_mat *  model);
    _shader->set_uniform("pointSize", 50.0f); // slightly larger points for visibility

    for(auto&& vao : _vert_arrays)
    {
        _shader->set_uniform("isPoint", vao.is_point ? 1.0f : 0.0f);
        vao.vert_array->draw();
    }

    return ;
}

auto ParticleRenderer::is_key_pressed(int  glfw_key) -> bool
{
    return glfwGetKey(_window.window, glfw_key) == GLFW_PRESS;
}