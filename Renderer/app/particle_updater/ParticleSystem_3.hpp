#ifndef NITROS_PARTICLE_SYSTEM_HPP
#define NITROS_PARTICLE_SYSTEM_HPP

#include "./particle_data.hpp"
#include "./cpu_/particle_updater_cpu.hpp"
#include "./gpu_/particle_updater_gpu.hpp"
#include "host_src/dev_memory.hpp"
#include "glcore/vertexarray.hpp"
#include "utilities/data/vecs.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif
#include <cuda_gl_interop.h>

namespace cudatr
{
    inline auto create_particle_updater(bool gpu_sim) -> std::unique_ptr<IParticleUpdater> {
        if( gpu_sim ) {
            return std::make_unique<ParticleUpdaterGPU>();
        }
        else {
            return std::make_unique<ParticleUpdaterCPU>();
        } 
    }

    class ParticleSystem
    {
    public:
        ParticleSystem(nitros::glcore::VertexArray* vertex_array, bool gpu_sim = true)
            : vert_array{vertex_array}
            , gpu_update_{gpu_sim }
            , part_updater{ create_particle_updater(gpu_sim) }
            , cuda_vbo_1{}
            , cuda_vbo_2{}
        {}

        void generate_grid(const nitros::utils::vec3f& bounds, float step);
        void generate_random(uint32_t count, const nitros::utils::vec3f& bounds);
        
        void update();
        void update_gpu(float dt);
        void update_cpu(float dt);

        [[nodiscard]] auto get_positions() const -> std::vector<nitros::utils::vec3f>;
        [[nodiscard]] auto size() const -> size_t { return part_updater->size(); }

        // Simulation statistics getters
        float get_temperature() const { return global_temp; }
        void set_temperature(float t) { global_temp = t; }
        float get_kinetic_temperature() const { return kinetic_temp; }
        float get_boundary() const { return container_boundary; }
        void set_boundary(float b) { container_boundary = b; sim_params.boundary = b; }
        unsigned int get_total_reactions() const { return total_reactions; }
        unsigned int get_total_collisions() const { return total_collisions; }
        float get_pressure() const;
        float get_reaction_rate() const { return reaction_rate; }
        float get_avg_kinetic_energy() const { return avg_ke; }
        unsigned int get_water_produced() const { return water_produced; }

    private:
        void update_gl_buffers(const std::vector<nitros::utils::vec3f>  &pos, const std::vector<nitros::utils::vec3f>  &col);

        nitros::glcore::VertexArray*    vert_array;
        bool                            gpu_update_;
        std::unique_ptr<IParticleUpdater>           part_updater;
        cudaGraphicsResource_t          cuda_vbo_1;
        cudaGraphicsResource_t          cuda_vbo_2;

        float                           global_temp = 300.0f;
        float                           kinetic_temp = 300.0f;
        float                           container_boundary = 2.2f;
        unsigned int                    total_reactions = 0;
        unsigned int                    total_collisions = 0;
        unsigned int                    water_produced = 0;
        float                           avg_ke = 0.0f;
        float                           reaction_rate = 0.0f;
        std::vector<unsigned int>       reaction_history;

        // Physics constants
        SimulationParams   sim_params = 
        {
            .gravity = -2.8f,
            .damping = 0.99f,
            .repulsion_radius = 0.15f,
            .repulsion_strength = 5.0f,
            .boundary = 2.2f,
            .bounce = 0.5f,
            .epsilon = 0.001f,
            .seed = 0,
            .temp = 300.0f,
            .collision_counter = nullptr,
            .reaction_counter = nullptr,
        };

        static auto get_random(float min, float max) -> float;
    };
}

#endif // NITROS_PARTICLE_SYSTEM_HPP
