#define NOMINMAX
/*----------------------------------------------------------------------------------------------------
Copyright © 2017-2020 Janakiraman Sankara Narayanan (johny.manasee@gmail.com). All Rights Reserved.
----------------------------------------------------------------------------------------------------*/

#include "renderer/window_wrapper.hpp"
#include "renderer/particle_renderer.hpp"

#include "utilities/rand/rand.hpp"
#include "utilities/data/vecs.hpp"
#include <glm/glm.hpp>

#include "renderer/objs/creator.hpp"
#include "renderer/transforms/transform.hpp"
// #include "camera/camera.hpp"

#include <iostream>
#include <thread>
#include <cstdint>
#include <random>
#include <memory>
#include <algorithm>
#include <cstdio>

#include "particle_updater/ParticleSystem_3.hpp"

namespace nitros
{
    auto generate_cube_lines(const utils::vec3f  &bounds) -> std::tuple< std::vector<utils::vec3f>, std::vector<utils::vec3f>, std::vector<utils::vec2Ui> >
    {
        float bx = bounds[0];
        float by = bounds[1];
        float bz = bounds[2];

        auto pos = std::vector<utils::vec3f> {
            {-bx,  by, -bz}, // 0: Back-Top-Left
            { bx,  by, -bz}, // 1: Back-Top-Right
            { bx, -by, -bz}, // 2: Back-Bottom-Right
            {-bx, -by, -bz}, // 3: Back-Bottom-Left

            {-bx,  by,  bz}, // 4: Front-Top-Left
            { bx,  by,  bz}, // 5: Front-Top-Right
            { bx, -by,  bz}, // 6: Front-Bottom-Right
            {-bx, -by,  bz}, // 7: Front-Bottom-Left
        };

        auto col = std::vector<utils::vec3f> {
            { 1, 0, 0}, // 0: Back-Top-Left
            { 1, 0, 0}, // 1: Back-Top-Right
            { 1, 0, 0}, // 2: Back-Bottom-Right
            { 1, 0, 0}, // 3: Back-Bottom-Left

            { 1, 0, 0}, // 4: Front-Top-Left
            { 1, 0, 0}, // 5: Front-Top-Right
            { 1, 0, 0}, // 6: Front-Bottom-Right
            { 1, 0, 0}, // 7: Front-Bottom-Left
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
    auto particle_renderer = ParticleRenderer{window};

    auto creator = Creator::get_instance();
    auto cube = creator.indexed_cube2({2, 2, 2}, {2, 2, 2});

    // Setting Mesh Object 1 (Wireframe boundary box)
    auto vao = std::make_unique<glcore::VertexArray>();
    auto [pos, col, ind] = generate_cube_lines({2.2f, 2.2f, 2.2f});
    vao->buffers[0] = [&]() 
    {
        auto buf_ = std::make_shared<glcore::Buffer>();
        buf_->write_data(pos);
        return buf_;
    }() ;
    vao->buffers[1] = [&]() 
    {
        auto buf_ = std::make_shared<glcore::Buffer>();
        buf_->write_data(col);
        return buf_;
    }() ;
    vao->index = [&]() 
    {
        auto buf_ = std::make_shared<glcore::Buffer>();
        buf_->write_data(ind);
        return buf_;
    }() ;
    vao->bind();
    vao->set_draw_mode(glcore::VertexArray::draw_mode::line);

    particle_renderer.insert_vertex_meshes(
        ParticleRenderer::Vert_Arr{
            .vert_array = std::move(vao),
            .is_point = false,
        } );

    // Setting Mesh Object 2 (Particles)
    auto vao_2 = std::make_unique<glcore::VertexArray>();
    auto particle_system = cudatr::ParticleSystem{vao_2.get(), true};

    auto rand_gen = [&ps = particle_system, vo= vao_2.get()]() 
    {
        ps.generate_random(15000, {1.2f, 1.2f, 1.2f});
        vo->bind();
        return ;
    };

    rand_gen();
    vao_2->set_draw_mode(glcore::VertexArray::draw_mode::points);
    particle_renderer.insert_vertex_meshes(
        ParticleRenderer::Vert_Arr{
            .vert_array = std::move(vao_2),
            .is_point = true,
        } );

    auto t1 = std::chrono::steady_clock::now();
    // loop
    window.run([&]()
    {   
        // Interactive Controls:
        // [T] / [Y]: Increase / Decrease global temperature
        if (particle_renderer.is_key_pressed(GLFW_KEY_T)) {
            float temp = particle_system.get_temperature();
            particle_system.set_temperature(temp + 5.0f);
        }
        if (particle_renderer.is_key_pressed(GLFW_KEY_Y)) {
            float temp = particle_system.get_temperature();
            particle_system.set_temperature(std::max(10.0f, temp - 5.0f));
        }

        // [P] / [O]: Compress / Expand container box
        if (particle_renderer.is_key_pressed(GLFW_KEY_P)) {
            float boundary = particle_system.get_boundary();
            float new_boundary = std::max(0.5f, boundary - 0.01f);
            particle_system.set_boundary(new_boundary);
            auto [new_pos, new_col, new_ind] = generate_cube_lines({new_boundary, new_boundary, new_boundary});
            auto* box_vao = particle_renderer.get_vert_arr(0)->vert_array.get();
            box_vao->buffers[0]->write_data(new_pos);
        }
        if (particle_renderer.is_key_pressed(GLFW_KEY_O)) {
            float boundary = particle_system.get_boundary();
            float new_boundary = std::min(3.0f, boundary + 0.01f);
            particle_system.set_boundary(new_boundary);
            auto [new_pos, new_col, new_ind] = generate_cube_lines({new_boundary, new_boundary, new_boundary});
            auto* box_vao = particle_renderer.get_vert_arr(0)->vert_array.get();
            box_vao->buffers[0]->write_data(new_pos);
        }

        // [R]: Reset only the temperature and boundary (pressure) parameters
        if (particle_renderer.is_key_pressed(GLFW_KEY_R)) {
            particle_system.set_temperature(300.0f);
            particle_system.set_boundary(2.2f);
            auto [new_pos, new_col, new_ind] = generate_cube_lines({2.2f, 2.2f, 2.2f});
            auto* box_vao = particle_renderer.get_vert_arr(0)->vert_array.get();
            box_vao->buffers[0]->write_data(new_pos);
        }

        // Rendering
        particle_renderer.render();

        // Updating
        particle_system.update();

        // Update stats title
        float dt_ms = nitros::Time2::delta_time().count();
        float fps = dt_ms > 0.0f ? 1000.0f / dt_ms : 0.0f;
        char title_buf[256];
        std::snprintf(title_buf, sizeof(title_buf),
            "3D H2-O2 Combustion | Temp (Act/Tgt): %.1f K / %.1f K | Press: %.1f Pa | Collisions: %u | Reactions: %u | H2O Produced: %u | FPS: %.1f",
            particle_system.get_kinetic_temperature(),
            particle_system.get_temperature(),
            particle_system.get_pressure(),
            particle_system.get_total_collisions(),
            particle_system.get_total_reactions(),
            particle_system.get_water_produced(),
            fps);
        glfwSetWindowTitle(window.window, title_buf);

        using namespace std::chrono_literals;
        std::this_thread::sleep_until(t1 + 10ms);
        t1 = std::chrono::steady_clock::now();
    });

    return 0;
}