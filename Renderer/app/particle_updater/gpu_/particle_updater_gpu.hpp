#pragma once

#include "../particle_data.hpp"
#include "../particle_updater.hpp"
#include "host_src/dev_memory.hpp"
#include <span>

namespace cudatr
{
    class ParticleUpdaterGPU final : public IParticleUpdater
    {
        private:
        GpuMemWrap_t<Particle>  _particles;
        GpuMemWrap_t<unsigned int> _collision_counter;
        GpuMemWrap_t<unsigned int> _reaction_counter;

        unsigned int _host_collisions = 0;
        unsigned int _host_reactions = 0;

        public:
        ParticleUpdaterGPU();
        ~ParticleUpdaterGPU();

        void init_particles(std::span<Particle>  particles) final;
        auto size() const -> std::size_t final;
        void update(OutResult  gpu_out_positions, const cudatr::SimulationParams  &sim_params, float dt) final;
        void get_particles(std::span<Particle> out_particles) final;
        unsigned int get_collision_count() const final { return _host_collisions; }
        unsigned int get_reaction_count() const final { return _host_reactions; }
    };
} // namespace nitros
