/*----------------------------------------------------------------------------------------------------
Copyright © 2017-2020 Janakiraman Sankara Narayanan (johny.manasee@gmail.com). All Rights Reserved.
----------------------------------------------------------------------------------------------------*/

#include "renderer/window_wrapper.hpp"
#include "renderer/shader_header.hpp"

#include "glcore/framebuffer.hpp"
#include "glcore/rasterizer.hpp"
#include "glcore/context.hpp"
#include "glcore/shader.h"
#include "glcore/vertexarray.hpp"

#include "utilities/rand/rand.hpp"
#include "utilities/data/vecs.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "renderer/objs/creator.hpp"
#include "renderer/transforms/transform.hpp"
// #include "camera/camera.hpp"

#include <iostream>
#include <thread>
#include <cstdint>
#include <random>
#include <memory>

#include "particle_updater/ParticleSystem_3.hpp"

namespace nitros
{
    auto get_shader_code() -> std::pair<std::string, std::string>
    {
        const auto vertex_shader1 =  R"( layout (location = 0)  in vec3 position;
                                    uniform mat4 mvp;
                                    out vec3 VColor;
                                    void main(void)
                                    {                
                                       gl_Position = mvp * vec4(position, 1.0);  
                                       VColor = vec3(1, 0, 0);
                                    } )";

        const auto frag_shader1 =   R"( in vec3 VColor;
                                    uniform vec4 m_col;
                                    out vec4 color;     
                                    void main(void) 
                                    {  
                                        // color = vec4(0.8, 0.8, 0.8, 1.0);  
                                        color = m_col;  
                                        // color = vec4(VColor, 1.0);
                                    } )";
                                    
        return {vertex_shader1, frag_shader1};
    }    

    auto generate_cube_lines(const utils::vec3f  &bounds) -> std::tuple< std::vector<utils::vec3f>, std::vector<utils::vec3f>, std::vector<utils::vec2Ui> >
    {
        auto pos = std::vector<utils::vec3f> {
            {-1, 1, -1}, // 0: Back-Top-Left
            { 1, 1, -1}, // 1: Back-Top-Right
            { 1,-1, -1}, // 2: Back-Bottom-Right
            {-1,-1, -1}, // 3: Back-Bottom-Left

            {-1, 1,  1}, // 4: Front-Top-Left
            { 1, 1,  1}, // 5: Front-Top-Right
            { 1,-1,  1}, // 6: Front-Bottom-Right
            {-1,-1,  1}, // 7: Front-Bottom-Left
        };

        auto col = std::vector<utils::vec3f> {
            { 1, 1, 1}, // 0: Back-Top-Left
            { 1, 1, 1}, // 1: Back-Top-Right
            { 1, 1, 1}, // 2: Back-Bottom-Right
            { 1, 1, 1}, // 3: Back-Bottom-Left

            { 1, 1, 1}, // 4: Front-Top-Left
            { 1, 1, 1}, // 5: Front-Top-Right
            { 1, 1, 1}, // 6: Front-Bottom-Right
            { 1, 1, 1}, // 7: Front-Bottom-Left
        };

        auto ins = std::vector<utils::vec2Ui> {
            // Back face
            {0, 1}, {1, 2}, {2, 3}, {3, 0},
            // Front face
            {4, 5}, {5, 6}, {6, 7}, {7, 4},
            // Connecting edges
            {0, 4}, {1, 5}, {2, 6}, {3, 7}
        };

        return {pos, col, ins};
    }
} // namespace nitros



int main(int argc, char const *argv[])
{
    auto window = Window{1920, 1080};

    using namespace nitros;
    using namespace std::string_literals;

    CUDA_CHECK( cudaSetDevice(0) );

    #if !defined(__EMSCRIPTEN__)
        glcore::load_context(glfwGetProcAddress);
    #endif
    {
        std::cout<< glcore::get_renderer() << std::endl;;
    
        {

            auto device_count = std::uint32_t{};
            auto devices = std::array<std::int32_t, 5>{};
        }
        
        // auto creator = Creator::get_instance();
        // auto cube = creator.indexed_cube2({2, 2, 2}, {2, 2, 2});

        auto vao = glcore::VertexArray{};
        auto [pos, col, ind] = generate_cube_lines({2, 2, 2});
        vao.buffers[0] = [&]() 
        {
            auto buf_ = std::make_shared<glcore::Buffer>();
            buf_->write_data(pos);
            return buf_;
        }() ;
        vao.buffers[1] = [&]() 
        {
            auto buf_ = std::make_shared<glcore::Buffer>();
            buf_->write_data(col);
            return buf_;
        }() ;
        vao.index = [&]() 
        {
            auto buf_ = std::make_shared<glcore::Buffer>();
            buf_->write_data(ind);
            return buf_;
        }() ;
        vao.bind();
        vao.set_draw_mode(glcore::VertexArray::draw_mode::line);

        auto vao_2 = glcore::VertexArray{};
        
        auto particle_system = cudatr::ParticleSystem{&vao_2, true};

        auto rand_gen = [&ps = particle_system, &vo= vao_2]() 
        {
            ps.generate_random(10'000, {0.8, 0.8, 0.8});
            vo.bind();
            return ;
        };

        rand_gen();
        vao_2.set_draw_mode(glcore::VertexArray::draw_mode::points);

        // Setting up Camera
        // auto cam = Camera{{0, 2, 5}};
        auto&& cam = window.camera;
        cam.Position = {0, 0, 4};

        auto [width, height] = window.window_dim();

        auto _fovy      = 45.0f;
        auto near_plane = 0.1f;
        auto far_plane  = 200.0f;
        auto projection = perspective_projection_matrix(_fovy, static_cast<float>(width), static_cast<float>(height), near_plane, far_plane);

        auto srt = SRT{ glm::vec3{1,1,1}, glm::vec3{0, 0, 0}, glm::vec3{0, 0, 0} };

        auto [vertex_shader1, frag_shader1] = get_shader_code();

        auto shader = glcore::Shader{vert_header + vertex_shader1, frag_header + frag_shader1};

        auto&& raster = glcore::Rasterizer::get_instance();
        raster.set_cull_face(glcore::Rasterizer::cull::front_and_back);
        raster.set_polygonMode(glcore::Rasterizer::polygonMode::fill);
        raster.set_point_size(5);

        auto t1 = std::chrono::steady_clock::now();
        // loop
        window.run([&]()
        {    
            auto&& frame_buffer = glcore::FrameBuffer::get_default();
            frame_buffer.bind();
            frame_buffer.clear_color({0.1f, 0.1, 0.1, 0.0f});
            frame_buffer.clear_depth();

            auto delta_time = Time2::delta_time();
            float dt = delta_time.count() / 1000.0f;

            srt.rotate.y += (glm::radians(10.f)*dt);
            auto model = model_matrix(srt);

            shader.use();
            auto view_mat = cam.GetViewMatrix();
            shader.set_uniform_matrix4fv("mvp", projection * view_mat *  model);
            shader.set_uniform("m_col", utils::vec4f{0.9, 0.0, 0.0, 1.f});
            vao.draw();
            shader.set_uniform("m_col", utils::vec4f{0.9, 0.9, 0.9, 1.f});
            vao_2.draw();

            static bool use_gpu = true;
            particle_system.update();
            
            if(glfwGetKey(window.window, GLFW_KEY_R) == GLFW_PRESS) {
                rand_gen();
            }

            using namespace std::chrono_literals;
            std::this_thread::sleep_until(t1 + 10ms);
            t1 = std::chrono::steady_clock::now();
            //std::cout<<"Frame Count "<<frame_count++<<std::endl;
        });
    }
    return 0;
}