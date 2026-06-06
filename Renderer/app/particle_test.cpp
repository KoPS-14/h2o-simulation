#include "ParticleSystem.hpp"
#include <iostream>
#include <vector>

int main() {
    using namespace nitros;
    
    ParticleSystem particle_system;
    particle_system.generate_grid({0.1f, 0.1f, 0.1f}, 0.1f);

    float dt = 0.016f; // ~60fps

    std::cout << "Starting Standalone Particle Test (Class Refactor)" << std::endl;
    for (int frame = 0; frame < 20; ++frame) {
        particle_system.update(dt);
        auto positions = particle_system.get_positions();
        std::cout << "Frame " << frame << ":" << std::endl;
        for (size_t i = 0; i < positions.size(); ++i) {
            std::cout << "  P" << i << ": pos(" << positions[i][0] << ", " << positions[i][1] << ", " << positions[i][2] << ")" << std::endl;
        }
    }

    return 0;
}
