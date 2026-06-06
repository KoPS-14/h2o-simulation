#include "./particle_updater_gpu.hpp"
#include <cstring>

extern "C" void launch_update_particles(cudatr::Particle* particles, nitros::utils::vec3f *out_positions, nitros::utils::vec3f  *out_colors, std::uint64_t count, cudatr::SimulationParams params, float dt);

namespace cudatr
{
    ParticleUpdaterGPU::ParticleUpdaterGPU()
        :IParticleUpdater{}
        ,_particles{create_gpu_mem_wrap<Particle>(0)}
        ,_collision_counter{create_gpu_mem_wrap<unsigned int>(1)}
        ,_reaction_counter{create_gpu_mem_wrap<unsigned int>(1)}
    {}
    ParticleUpdaterGPU::~ParticleUpdaterGPU() = default;

    void ParticleUpdaterGPU::init_particles(std::span<Particle>  particles) 
    {
        _particles->copy_from_host_to_device( particles );
        // Reset counters
        unsigned int zero = 0;
        CUDA_CHECK( cudaMemcpy(_collision_counter->gpu_ptr(), &zero, sizeof(unsigned int), cudaMemcpyHostToDevice) );
        CUDA_CHECK( cudaMemcpy(_reaction_counter->gpu_ptr(), &zero, sizeof(unsigned int), cudaMemcpyHostToDevice) );
    }

    auto ParticleUpdaterGPU::size() const -> std::size_t  
    {
        return _particles->count();
    }

    void ParticleUpdaterGPU::update(OutResult  gpu_out_positions, const cudatr::SimulationParams  &sim_params, float dt)
    {
        // Pass device counter pointers via a mutable copy of sim_params
        cudatr::SimulationParams params_copy = sim_params;
        params_copy.collision_counter = _collision_counter->gpu_ptr();
        params_copy.reaction_counter  = _reaction_counter->gpu_ptr();

        launch_update_particles(_particles->gpu_ptr(), gpu_out_positions.out_positions.data(), gpu_out_positions.out_colors.data(), static_cast<std::uint64_t>( _particles->count() ), params_copy, dt);
        CUDA_CHECK( cudaDeviceSynchronize() );

        // Copy counters back
        CUDA_CHECK( cudaMemcpy(&_host_collisions, _collision_counter->gpu_ptr(), sizeof(unsigned int), cudaMemcpyDeviceToHost) );
        CUDA_CHECK( cudaMemcpy(&_host_reactions, _reaction_counter->gpu_ptr(), sizeof(unsigned int), cudaMemcpyDeviceToHost) );
    }

    void ParticleUpdaterGPU::get_particles(std::span<Particle> out_particles)
    {
        _particles->copy_from_device_to_host(out_particles);
    }
} // namespace cudatr

