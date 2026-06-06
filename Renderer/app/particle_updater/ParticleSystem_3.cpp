#include "ParticleSystem_3.hpp"
#include <random>
#include <cmath>
#include <memory>
#include <iostream>

namespace cudatr
{
    void ParticleSystem::generate_grid(const nitros::utils::vec3f& bounds, float step)
    {
        auto particles = std::vector<cudatr::Particle>{};

        for(auto i = -bounds[0] ; i <= bounds[0]; i += step)
        {
            for(auto j = -bounds[1] ; j <= bounds[1]; j += step)
            {
                for(auto k = -bounds[2] ; k <= bounds[2]; k += step)
                {
                    particles.push_back(
                        Particle
                        {
                            .position = {i, j, k},
                            .color    = {1, 1, 1},
                            .velocity = glm::vec3{0.f} 
                        });
                }
            }
        }

        // part_updater_cpu.init_particles(particles);
        part_updater->init_particles(particles);

        auto positions = std::vector<nitros::utils::vec3f>{};
        positions.reserve(particles.size());
        for (const auto& p : particles) {
            positions.push_back(p.position);
        }

        auto colors = std::vector<nitros::utils::vec3f>{};
        colors.reserve(particles.size());
        for (const auto& p : particles){
            colors.push_back(p.color);
        } 

        update_gl_buffers( positions, colors );
    }

    void ParticleSystem::generate_random(uint32_t count, const nitros::utils::vec3f& bounds)
    {
        uint32_t h2_count = 10000;
        uint32_t o2_count = 5000;
        uint32_t total_count = h2_count + o2_count;

        // Reset statistics on host when generating/resetting simulation
        total_reactions = 0;
        total_collisions = 0;
        water_produced = 0;
        avg_ke = 0.0f;
        reaction_rate = 0.0f;
        kinetic_temp = global_temp;
        reaction_history.clear();

        auto particles = std::vector<cudatr::Particle>{};
        particles.reserve(total_count);

        // Helper to generate Gaussian random numbers for Maxwell-Boltzmann distribution
        auto generate_gaussian = [&](float mean, float stddev) -> float {
            float u1 = get_random(0.001f, 1.0f);
            float u2 = get_random(0.001f, 1.0f);
            return mean + stddev * std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265f * u2);
        };

        // Standard deviations of velocity for mass H2=2.0 and O2=32.0 at T = 300K
        // Scaled up by 2.0 to represent slightly faster real-world gas velocities
        float sigma_h2 = 2.0f * std::sqrt(global_temp / 300.0f) * (0.565f / std::sqrt(2.0f));
        float sigma_o2 = 2.0f * std::sqrt(global_temp / 300.0f) * (0.565f / std::sqrt(32.0f));

        // Generate H2 (Red)
        for (uint32_t i = 0; i < h2_count; ++i)
        {
            particles.push_back(Particle{
                .position = {
                    get_random(-bounds[0], bounds[0]),
                    get_random(-bounds[1], bounds[1]),
                    get_random(-bounds[2], bounds[2])
                },
                .color = {1.0f, 0.0f, 0.0f}, // Red
                .velocity = glm::vec3{
                    generate_gaussian(0.0f, sigma_h2),
                    generate_gaussian(0.0f, sigma_h2),
                    generate_gaussian(0.0f, sigma_h2)
                },
                .type = 1 // TYPE_H2
            });
        }

        // Generate O2 (Green)
        for (uint32_t i = 0; i < o2_count; ++i)
        {
            particles.push_back(Particle{
                .position = {
                    get_random(-bounds[0], bounds[0]),
                    get_random(-bounds[1], bounds[1]),
                    get_random(-bounds[2], bounds[2])
                },
                .color = {0.0f, 1.0f, 0.0f}, // Green
                .velocity = glm::vec3{
                    generate_gaussian(0.0f, sigma_o2),
                    generate_gaussian(0.0f, sigma_o2),
                    generate_gaussian(0.0f, sigma_o2)
                },
                .type = 2 // TYPE_O2
            });
        }

        // part_updater_cpu.init_particles(particles);
        part_updater->init_particles(particles);

        auto positions = std::vector<nitros::utils::vec3f>{};
        positions.reserve(particles.size());
        auto colors = std::vector<nitros::utils::vec3f>{};
        colors.reserve(particles.size());
        for (const auto& p : particles){
            positions.push_back(p.position);
            colors.push_back(p.color);
        } 
        
        update_gl_buffers(positions, colors);
    }

    void ParticleSystem::update_gl_buffers(const std::vector<nitros::utils::vec3f>  &positions, const std::vector<nitros::utils::vec3f>  &colors)
    {
        auto buf_ = std::make_shared<nitros::glcore::Buffer>();
        buf_->write_data(positions);
        vert_array->buffers[0] = buf_;

        auto buf_2 = std::make_shared<nitros::glcore::Buffer>();
        buf_2->write_data(colors);
        vert_array->buffers[1] = buf_2;

        vert_array->bind();

        CUDA_CHECK( cudaGraphicsGLRegisterBuffer(&cuda_vbo_1, buf_->get_id() , cudaGraphicsRegisterFlagsWriteDiscard) );
        CUDA_CHECK( cudaGraphicsGLRegisterBuffer(&cuda_vbo_2, buf_2->get_id(), cudaGraphicsRegisterFlagsWriteDiscard) );
    }

    void ParticleSystem::update_gpu(float dt)
    {
        CUDA_CHECK( cudaGraphicsMapResources(1, &cuda_vbo_1, 0) );
        CUDA_CHECK( cudaGraphicsMapResources(1, &cuda_vbo_2, 0) );
        
        void* cuda_vbo_pos_data;
        size_t cuda_vbo_pos_size;
        CUDA_CHECK( cudaGraphicsResourceGetMappedPointer(&cuda_vbo_pos_data, &cuda_vbo_pos_size, cuda_vbo_1) );

        void* cuda_vbo_col_data;
        size_t cuda_vbo_col_size;
        CUDA_CHECK( cudaGraphicsResourceGetMappedPointer(&cuda_vbo_col_data, &cuda_vbo_col_size, cuda_vbo_2) );
        
        assert(cuda_vbo_pos_size == part_updater->size() * sizeof(nitros::utils::vec3f) );

        auto out_result = OutResult
        {
            .out_positions = std::span<nitros::utils::vec3f>{ static_cast<nitros::utils::vec3f*>(cuda_vbo_pos_data), cuda_vbo_pos_size },
            .out_colors    = std::span<nitros::utils::vec3f>{ static_cast<nitros::utils::vec3f*>(cuda_vbo_col_data), cuda_vbo_col_size },
        };

        part_updater->update( std::move(out_result), sim_params, dt );

        CUDA_CHECK( cudaGraphicsUnmapResources(1, &cuda_vbo_1, 0) );
        CUDA_CHECK( cudaGraphicsUnmapResources(1, &cuda_vbo_2, 0) );
    }

    void ParticleSystem::update()
    {
        sim_params.seed++;
        sim_params.temp = global_temp;
        sim_params.boundary = container_boundary;

        float dt = 0.016f;
        if(gpu_update_) {
            update_gpu(dt);
        }
        else {
            update_cpu(dt);
        }

        // Retrieve particle array to CPU to calculate stats
        std::vector<Particle> host_particles(part_updater->size());
        part_updater->get_particles(host_particles);

        double total_ke = 0.0;
        unsigned int active_gases = 0;
        unsigned int water_count = 0;
        unsigned int active_count = 0;

        for (const auto& p : host_particles)
        {
            if (p.type == 0) continue; // Inactive
            active_count++;

            float mass = 2.0f;
            if (p.type == 1) {
                mass = 2.0f; // H2
                active_gases++;
            } else if (p.type == 2) {
                mass = 32.0f; // O2
                active_gases++;
            } else if (p.type == 3 || p.type == 4) {
                mass = 18.0f; // H2O
                water_count++;
                if (p.type == 4) {
                    active_gases++; // Water vapor is an active gas
                }
            }

            float speedSq = p.velocity.x * p.velocity.x + p.velocity.y * p.velocity.y + p.velocity.z * p.velocity.z;
            total_ke += 0.5f * mass * speedSq;
        }

        water_produced = water_count;
        if (active_count > 0) {
            avg_ke = total_ke / active_count;
        } else {
            avg_ke = 0.0f;
        }

        // Temperature calculated from active gas kinetic energy
        if (active_gases > 0) {
            float calculated_temp = (total_ke / active_gases) * 12.0f;
            kinetic_temp = 0.98f * kinetic_temp + 0.02f * calculated_temp;
        } else {
            kinetic_temp = global_temp;
        }

        total_reactions = part_updater->get_reaction_count();
        total_collisions = part_updater->get_collision_count();

        // Calculate reaction rate (reactions per second)
        reaction_history.push_back(total_reactions);
        if (reaction_history.size() > 60) {
            unsigned int diff = total_reactions - reaction_history.front();
            reaction_history.erase(reaction_history.begin());
            reaction_rate = (float)diff / (60.0f * dt);
        } else {
            reaction_rate = (float)total_reactions / (reaction_history.size() * dt);
        }
    }

    float ParticleSystem::get_pressure() const
    {
        // P = n * R * T / V
        float V = std::pow(2.0f * container_boundary, 3);
        unsigned int gas_count = part_updater->size() - water_produced;
        float rho = (float)gas_count / V;
        return rho * 1.824f * global_temp; // scaled to Pa
    }

    void ParticleSystem::update_cpu(float dt)
    {
        auto particles = std::vector<nitros::utils::vec3f>( part_updater->size() );
        auto colors = std::vector<nitros::utils::vec3f>( part_updater->size() );

        auto out_result = OutResult{
            .out_positions = particles,
            .out_colors = colors
        };

        part_updater->update( std::move(out_result), sim_params, dt );

        vert_array->buffers[0]->write_data(particles);
        vert_array->buffers[1]->write_data(colors);
        vert_array->bind();
    }

    auto ParticleSystem::get_positions() const -> std::vector<nitros::utils::vec3f>
    {
        return {}; 
    }

    auto ParticleSystem::get_random(float min, float max) -> float
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(min, max);
        return dis(gen);
    }
}

