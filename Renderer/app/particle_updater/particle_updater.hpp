#ifndef NITROS_PARTICLE_UPDATER_HPP
#define NITROS_PARTICLE_UPDATER_HPP

#include "./particle_data.hpp"
#include <span>

namespace cudatr
{
    struct OutResult
    {
        std::span<nitros::utils::vec3f>  out_positions;
        std::span<nitros::utils::vec3f>  out_colors;
    };

    class IParticleUpdater
    {
        public:
        IParticleUpdater() = default;
        virtual ~IParticleUpdater() = default;

        virtual void init_particles(std::span<Particle>  particles) = 0;
        virtual auto size() const -> std::size_t = 0;
        virtual void update(OutResult  out_positions, const cudatr::SimulationParams  &sim_params, float dt) = 0;
        virtual void get_particles(std::span<Particle> out_particles) = 0;
        virtual unsigned int get_collision_count() const = 0;
        virtual unsigned int get_reaction_count() const = 0;
    };
} // namespace cudatr


#endif