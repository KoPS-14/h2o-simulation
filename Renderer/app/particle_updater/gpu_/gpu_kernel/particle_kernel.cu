#include "../../particle_data.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <span>
#include <iostream>
#include <stdio.h>

namespace cudatr
{
    __device__ inline uint32_t xorshift32(uint32_t *state) {
        uint32_t x = *state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *state = x;
        return x;
    }

    __device__ inline float get_radius(int type) {
        if (type == 1) return 0.015f; // H2
        if (type == 2) return 0.022f; // O2
        if (type == 3) return 0.019f; // H2O liquid
        if (type == 4) return 0.019f; // H2O vapor
        return 0.0f;
    }

    __device__ inline float get_mass(int type) {
        if (type == 1) return 2.0f;  // H2
        if (type == 2) return 32.0f; // O2
        if (type == 3) return 18.0f; // H2O liquid
        if (type == 4) return 18.0f; // H2O vapor
        return 1.0f;
    }

    __global__ void update_particles_kernel(Particle* particles, nitros::utils::vec3f *out_positions, nitros::utils::vec3f  *out_colors, std::uint64_t count, SimulationParams params, float dt)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;

        Particle& p = particles[i];

        // 1. Inactive particle handling
        if (p.type == 0) // TYPE_INACTIVE
        {
            p.position = nitros::utils::vec3f{9999.0f, 9999.0f, 9999.0f};
            p.velocity = glm::vec3(0.0f);
            out_positions[i] = p.position;
            out_colors[i] = nitros::utils::vec3f{0.0f, 0.0f, 0.0f};
            return;
        }

        // Fade out spark/flash colors for H2O molecules and transition between vapor (grey) and liquid (blue)
        if (p.type == 3 || p.type == 4)
        {
            glm::vec3 target_color = (p.type == 4) ? glm::vec3(0.6f, 0.6f, 0.6f) : glm::vec3(0.0f, 0.0f, 1.0f);
            float decay_rate = 5.0f * dt;
            for (int c = 0; c < 3; ++c) {
                if (p.color[c] < target_color[c]) {
                    p.color[c] += decay_rate;
                    if (p.color[c] > target_color[c]) p.color[c] = target_color[c];
                } else if (p.color[c] > target_color[c]) {
                    p.color[c] -= decay_rate;
                    if (p.color[c] < target_color[c]) p.color[c] = target_color[c];
                }
            }
        }

        // Phase transition for H2O (evaporation/condensation)
        if (p.type == 3 || p.type == 4)
        {
            uint32_t rand_state = i * 1664525u + 1013904223u + params.seed;
            // Height-based temperature gradient: cooler at the top, warmer at the bottom
            float local_temp = params.temp * (1.0f - 0.15f * (p.position[1] / params.boundary));
            if (local_temp < 10.0f) local_temp = 10.0f;

            if (p.type == 3 && local_temp >= 373.15f)
            {
                // Evaporation probability factors:
                // 1. Temperature: (local_temp - 373.15) / 100.0f
                float temp_factor = (local_temp - 373.15f) / 100.0f;

                // 2. Available surface area: liquid particles closer to the top of the pool evaporate faster!
                float height_factor = (p.position[1] + params.boundary) / (2.0f * params.boundary);
                if (height_factor < 0.0f) height_factor = 0.0f;
                if (height_factor > 1.0f) height_factor = 1.0f;

                // 3. Local energy: higher speed increases evaporation chance
                float speedSq = p.velocity[0]*p.velocity[0] + p.velocity[1]*p.velocity[1] + p.velocity[2]*p.velocity[2];
                float energy_factor = 1.0f + 0.05f * speedSq;

                // Evaporation chance: base chance 0.5% scaled by temp, surface height, and local kinetic energy
                float base_chance = 0.005f;
                float evaporation_chance = base_chance * temp_factor * (height_factor * 0.8f + 0.2f) * energy_factor;

                float r = (float)(xorshift32(&rand_state)) / 4294967296.0f;
                if (r < evaporation_chance)
                {
                    p.type = 4;
                    p.velocity[1] += 1.5f; // Evaporated steam rises
                }
            }
            else if (p.type == 4 && local_temp < 373.15f)
            {
                // Condensation probability factors:
                // 1. Lower temperature: (373.15 - local_temp) / 50.0f
                float temp_factor = (373.15f - local_temp) / 50.0f;

                // 2. Contact with liquid water (nucleation) and local steam density
                bool near_liquid = false;
                int nearby_vapor = 0;
                float search_radius = 0.15f;
                for (std::uint64_t j = 0; j < count; ++j)
                {
                    if (j == i) continue;
                    int o_type = particles[j].type;
                    if (o_type == 3) // liquid H2O
                    {
                        glm::vec3 diff = {p.position[0] - particles[j].position[0], p.position[1] - particles[j].position[1], p.position[2] - particles[j].position[2]};
                        float distSq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                        if (distSq < search_radius * search_radius)
                        {
                            near_liquid = true;
                        }
                    }
                    else if (o_type == 4) // vapor H2O
                    {
                        glm::vec3 diff = {p.position[0] - particles[j].position[0], p.position[1] - particles[j].position[1], p.position[2] - particles[j].position[2]};
                        float distSq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                        if (distSq < search_radius * search_radius)
                        {
                            nearby_vapor++;
                        }
                    }
                }

                // Condensation chance: base chance 3%, increases to 8% near liquid water, scaled by temp and vapor density
                float vapor_density_factor = 1.0f + 0.1f * (float)nearby_vapor;
                float base_chance = near_liquid ? 0.08f : 0.03f;
                float condensation_chance = base_chance * temp_factor * vapor_density_factor;

                float r = (float)(xorshift32(&rand_state)) / 4294967296.0f;
                if (r < condensation_chance)
                {
                    p.type = 3;
                    p.velocity *= 0.5f; // Particle loses energy when condensing to join nearby droplets
                }
            }
        }

        // 2. Chemical reaction check (O2 searches for two H2)
        if (p.type == 2) // TYPE_O2
        {
            float rxn_radius = 0.08f; // increased from 0.06f to increase reaction rate
            int idx1 = -1;
            int idx2 = -1;

            for (std::uint64_t j = 0; j < count; ++j)
            {
                if (i == j) continue;
                const Particle& other = particles[j];
                if (other.type == 1) // H2
                {
                    glm::vec3 diff = {p.position[0] - other.position[0], p.position[1] - other.position[1], p.position[2] - other.position[2]};
                    float distSq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                    if (distSq < rxn_radius * rxn_radius)
                    {
                        if (idx1 == -1) {
                            idx1 = j;
                        } else {
                            idx2 = j;
                            break; // Found two H2 candidates
                        }
                    }
                }
            }

            if (idx1 != -1 && idx2 != -1)
            {
                // Calculate relative kinetic energy
                // E_rel = 0.5 * mu * v_rel^2 for each H2 relative to O2
                // mu_H2_O2 = (2.0 * 32.0) / (2.0 + 32.0) = 1.882f
                glm::vec3 v_rel1 = particles[idx1].velocity - p.velocity;
                glm::vec3 v_rel2 = particles[idx2].velocity - p.velocity;
                float E_rel = 0.941f * (v_rel1.x * v_rel1.x + v_rel1.y * v_rel1.y + v_rel1.z * v_rel1.z +
                                        v_rel2.x * v_rel2.x + v_rel2.y * v_rel2.y + v_rel2.z * v_rel2.z);

                float E_a_kinetic = 1.0f; // lowered from 3.0f to increase reaction rate
                if (E_rel > E_a_kinetic)
                {
                    // Arrhenius rate: P = exp(-E_a / T)
                    float E_a_arrhenius = 550.0f; // lowered from 800.0f to increase reaction rate
                    float P = std::exp(-E_a_arrhenius / params.temp);

                    uint32_t rand_state = i * 1664525u + 1013904223u + params.seed;
                    float r = (float)(xorshift32(&rand_state)) / 4294967296.0f;

                    if (r < P)
                    {
                        // Try to claim both H2 molecules
                        int old1 = atomicCAS(&(particles[idx1].type), 1, 0);
                        if (old1 == 1)
                        {
                            int old2 = atomicCAS(&(particles[idx2].type), 1, 0);
                            if (old2 == 1)
                            {
                                // Success! React!
                                // Center of mass velocity
                                glm::vec3 v_cm = (32.0f * p.velocity + 2.0f * particles[idx1].velocity + 2.0f * particles[idx2].velocity) / 36.0f;

                                // Generate direction for emission
                                float theta = ((float)(xorshift32(&rand_state)) / 4294967296.0f) * 2.0f * 3.14159265f;
                                float z_val = ((float)(xorshift32(&rand_state)) / 4294967296.0f) * 2.0f - 1.0f;
                                float r_xy = std::sqrt(1.0f - z_val * z_val);
                                glm::vec3 u = {r_xy * std::cos(theta), r_xy * std::sin(theta), z_val};

                                // Exothermic heat release: boost velocity of the two H2O molecules (low boost so they drop straight down)
                                float v_boost = 0.5f; 
                                p.velocity = v_cm + v_boost * u;
                                particles[idx1].velocity = v_cm - v_boost * u;

                                // Boost velocity of nearby molecules (shockwave / heat release)
                                float heat_radius = 0.25f;
                                float heat_boost_strength = 4.0f;
                                for (std::uint64_t k = 0; k < count; ++k)
                                {
                                    if (k == i || k == idx1 || k == idx2) continue;
                                    Particle& near_p = particles[k];
                                    if (near_p.type == 0) continue;

                                    glm::vec3 to_near = {near_p.position[0] - p.position[0], near_p.position[1] - p.position[1], near_p.position[2] - p.position[2]};
                                    float d2 = to_near.x * to_near.x + to_near.y * to_near.y + to_near.z * to_near.z;
                                    if (d2 < heat_radius * heat_radius && d2 > 0.0001f)
                                    {
                                        float d = std::sqrt(d2);
                                        float factor = 1.0f - d / heat_radius;
                                        near_p.velocity += (to_near / d) * heat_boost_strength * factor;
                                    }
                                }

                                // Convert O2 -> H2O (start as bright spark/flash)
                                p.type = 3;
                                p.color = {3.0f, 3.0f, 1.0f}; // Bright yellow/white flash

                                // Convert first H2 -> H2O (start as bright spark/flash)
                                particles[idx1].type = 3;
                                particles[idx1].color = {3.0f, 3.0f, 1.0f}; // Bright yellow/white flash

                                // The second H2 remains type 0 (inactive), representing stoichiometry

                                atomicAdd(params.reaction_counter, 1);
                            }
                            else
                            {
                                // Release first H2 back
                                atomicCAS(&(particles[idx1].type), 0, 1);
                            }
                        }
                    }
                }
            }
        }

        // 3. Elastic sphere collisions between all active molecules
        float r_i = get_radius(p.type);
        float m_i = get_mass(p.type);
        for (std::uint64_t j = 0; j < count; ++j)
        {
            if (i == j) continue;
            const Particle& other = particles[j];
            if (other.type == 0) continue;

            float r_j = get_radius(other.type);
            float m_j = get_mass(other.type);
            float R_sum = r_i + r_j;

            glm::vec3 diff = {p.position[0] - other.position[0], p.position[1] - other.position[1], p.position[2] - other.position[2]};
            float distSq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
            if (distSq < R_sum * R_sum && distSq > 0.000001f)
            {
                float dist = std::sqrt(distSq);
                glm::vec3 n = diff / dist; // normal from other to p

                // Relative velocity
                glm::vec3 v_rel = p.velocity - other.velocity;
                float v_normal = v_rel.x * n.x + v_rel.y * n.y + v_rel.z * n.z;

                if (v_normal < 0.0f) // moving towards each other
                {
                    float e = 0.95f; // high restitution
                    if (p.type == 3 || other.type == 3) {
                        e = 0.05f; // liquid water collisions are highly inelastic
                    }
                    float J = -(1.0f + e) * v_normal / (1.0f / m_i + 1.0f / m_j);
                    p.velocity += (J / m_i) * n;

                    // Increment collision counter (only once per pair)
                    if (i < j)
                    {
                        atomicAdd(params.collision_counter, 1);
                    }
                }

                // Position correction to prevent overlap (push-apart)
                float penetration = R_sum - dist;
                p.position[0] += 0.5f * penetration * n.x;
                p.position[1] += 0.5f * penetration * n.y;
                p.position[2] += 0.5f * penetration * n.z;
            }
        }

        // 4. Water condensation attraction (cohesion - only for liquid water)
        if (p.type == 3)
        {
            float cohesion_radius = 0.25f;
            float cohesion_strength = 0.8f; // Gentle attraction strength so it pools slowly
            for (std::uint64_t j = 0; j < count; ++j)
            {
                if (i == j) continue;
                const Particle& other = particles[j];
                if (other.type != 3) continue; // Only attract to other H2O

                glm::vec3 diff = {p.position[0] - other.position[0], p.position[1] - other.position[1], p.position[2] - other.position[2]};
                float distSq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                if (distSq < cohesion_radius * cohesion_radius && distSq > 0.0001f)
                {
                    float dist = std::sqrt(distSq);
                    // Attractive force towards the other molecule (opposite of repulsion)
                    glm::vec3 force = -(diff / dist) * cohesion_strength * (1.0f - dist / cohesion_radius);
                    p.velocity += force * dt;
                }
            }
        }

        // 5. Apply forces and physical attributes based on type
        float local_temp = params.temp * (1.0f - 0.15f * (p.position[1] / params.boundary));
        if (local_temp < 10.0f) local_temp = 10.0f;

        float gravity = 0.0f;
        float damping = params.damping;
        float bounce = params.bounce;
        bool has_thermal = false;

        if (p.type == 1 || p.type == 2) // H2 or O2
        {
            gravity = 0.0f; // Gases float / diffuse
            damping = 0.999f; // Low damping to keep gases energetic
            bounce = 1.0f; // Elastic boundary collisions
            has_thermal = true; // Brownian motion
        }
        else if (p.type == 3) // Liquid H2O
        {
            gravity = params.gravity * 2.0f; // Falls down under gravity
            damping = 0.90f; // Stronger damping to make it settle
            bounce = 0.0f; // No bounce on container boundaries
            has_thermal = false; // Liquid, no thermal gas random walk

            // If liquid water is at the bottom, stop its movement completely
            float limit = params.boundary - params.epsilon;
            if (p.position[1] <= -limit + 0.08f)
            {
                damping = 0.05f; // Extremely high damping to make it stay still
                p.velocity[0] = 0.0f;
                p.velocity[2] = 0.0f;
                if (p.velocity[1] < 0.0f) p.velocity[1] = 0.0f;
            }
        }
        else if (p.type == 4) // Vapor H2O
        {
            gravity = 0.0f; // Vapor floats
            damping = 0.999f; // Low damping
            bounce = 1.0f; // Elastic bounce
            has_thermal = true; // Gas random walk

            // Apply buoyancy: hot steam rises naturally
            float buoyancy = 0.8f * (local_temp / 300.0f);
            p.velocity[1] += buoyancy * dt;
        }

        // Apply gravity
        p.velocity[1] += gravity * dt;

        // Apply thermal motion (Brownian motion) for air molecules
        if (has_thermal)
        {
            // Seed PRNG using thread ID and host seed
            uint32_t rand_state = i * 1664525u + 1013904223u + params.seed;
            float rx = ((float)(xorshift32(&rand_state)) / 4294967296.0f) * 2.0f - 1.0f;
            float ry = ((float)(xorshift32(&rand_state)) / 4294967296.0f) * 2.0f - 1.0f;
            float rz = ((float)(xorshift32(&rand_state)) / 4294967296.0f) * 2.0f - 1.0f;

            // Thermal speed scales with square root of local temperature
            float thermal_force = 1.0f * std::sqrt(local_temp / 300.0f);
            p.velocity[0] += rx * thermal_force * dt;
            p.velocity[1] += ry * thermal_force * dt;
            p.velocity[2] += rz * thermal_force * dt;
        }

        // Update position
        p.position[0] += p.velocity[0] * dt;
        p.position[1] += p.velocity[1] * dt;
        p.position[2] += p.velocity[2] * dt;

        // Boundary collisions
        float limit = params.boundary - params.epsilon;
        if (p.position[0] > limit) { p.position[0] = limit; if (p.velocity[0] > 0) p.velocity[0] *= -bounce; }
        if (p.position[0] < -limit) { p.position[0] = -limit; if (p.velocity[0] < 0) p.velocity[0] *= -bounce; }
        if (p.position[1] > limit) { p.position[1] = limit; if (p.velocity[1] > 0) p.velocity[1] *= -bounce; }
        if (p.position[1] < -limit) { p.position[1] = -limit; if (p.velocity[1] < 0) p.velocity[1] *= -bounce; }
        if (p.position[2] > limit) { p.position[2] = limit; if (p.velocity[2] > 0) p.velocity[2] *= -bounce; }
        if (p.position[2] < -limit) { p.position[2] = -limit; if (p.velocity[2] < 0) p.velocity[2] *= -bounce; }

        // Damping
        p.velocity *= damping;

        // Write to VBO outputs
        out_positions[i] = p.position;
        out_colors[i]    = p.color;
    }
}

extern "C" void launch_update_particles(cudatr::Particle* particles, nitros::utils::vec3f *out_pos, nitros::utils::vec3f *out_cols, std::uint64_t count, cudatr::SimulationParams params, float dt)
{
    if (count == 0) return;
    int threadsPerBlock = 1024;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    cudatr::update_particles_kernel<<<blocksPerGrid, threadsPerBlock>>>(particles, out_pos, out_cols, count, params, dt);
}

