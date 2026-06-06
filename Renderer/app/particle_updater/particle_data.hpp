#ifndef NITROS_PARTICLE_DATA_HPP
#define NITROS_PARTICLE_DATA_HPP

#include "utilities/data/vecs.hpp"
#include <glm/glm.hpp>

namespace cudatr
{
    struct Particle
    {
        nitros::utils::vec3f position;
        nitros::utils::vec3f color;
        glm::vec3 velocity;
        int type; // 0 = inactive, 1 = H2, 2 = O2, 3 = H2O
    };

    struct SimulationParams
    {
        float dt;
        float gravity;
        float damping;
        float repulsion_radius;
        float repulsion_strength;
        float boundary;
        float bounce;
        float epsilon;
        unsigned int seed;
        float temp;
        unsigned int* collision_counter;
        unsigned int* reaction_counter;
    };
};

#endif
