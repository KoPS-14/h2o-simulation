#pragma once

#include "../particle_data.hpp"
#include "../particle_updater.hpp"
#include <span>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace cudatr
{
    class ParticleUpdaterCPU final : public IParticleUpdater
    {
        private:
        std::vector<Particle>   _particles;
        unsigned int _host_collisions = 0;
        unsigned int _host_reactions = 0;

        float get_radius(int type) const {
            if (type == 1) return 0.015f; // H2
            if (type == 2) return 0.022f; // O2
            if (type == 3) return 0.019f; // H2O liquid
            if (type == 4) return 0.019f; // H2O vapor
            return 0.0f;
        }

        float get_mass(int type) const {
            if (type == 1) return 2.0f;  // H2
            if (type == 2) return 32.0f; // O2
            if (type == 3) return 18.0f; // H2O liquid
            if (type == 4) return 18.0f; // H2O vapor
            return 1.0f;
        }

        float rand_float() {
            return (float)std::rand() / RAND_MAX;
        }

        public:
        ParticleUpdaterCPU()
            :IParticleUpdater{}
            ,_particles{}
        {}
        ~ParticleUpdaterCPU()
        {}

        void init_particles(std::span<Particle>  particles) final
        {
            _particles.resize(particles.size());
            std::copy( particles.begin(), particles.end(), _particles.begin() );
            _host_collisions = 0;
            _host_reactions = 0;
        }

        auto size() const -> std::size_t final 
        {
            return _particles.size();
        }

        void update(OutResult  out_result, const cudatr::SimulationParams  &sim_params, float dt) final
        {
            auto&& out_positions = out_result.out_positions;
            auto&& out_colors = out_result.out_colors;
            assert(out_positions.size() >= _particles.size());
            assert(out_colors.size() >= _particles.size());

            size_t count = _particles.size();

            // Fade out spark/flash colors for H2O molecules and transition between vapor (grey) and liquid (blue)
            for (size_t i = 0; i < count; ++i)
            {
                Particle& p = _particles[i];
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
            }

            // Phase transition for H2O (evaporation/condensation)
            for (size_t i = 0; i < count; ++i)
            {
                Particle& p = _particles[i];
                if (p.type == 3 || p.type == 4)
                {
                    // Height-based temperature gradient: cooler at the top, warmer at the bottom
                    float local_temp = sim_params.temp * (1.0f - 0.15f * (p.position[1] / sim_params.boundary));
                    if (local_temp < 10.0f) local_temp = 10.0f;

                    if (p.type == 3 && local_temp >= 373.15f)
                    {
                        // Evaporation probability factors:
                        // 1. Temperature: (local_temp - 373.15) / 100.0f
                        float temp_factor = (local_temp - 373.15f) / 100.0f;

                        // 2. Available surface area: liquid particles closer to the top of the pool evaporate faster!
                        float height_factor = (p.position[1] + sim_params.boundary) / (2.0f * sim_params.boundary);
                        if (height_factor < 0.0f) height_factor = 0.0f;
                        if (height_factor > 1.0f) height_factor = 1.0f;

                        // 3. Local energy: higher speed increases evaporation chance
                        float speedSq = p.velocity.x * p.velocity.x + p.velocity.y * p.velocity.y + p.velocity.z * p.velocity.z;
                        float energy_factor = 1.0f + 0.05f * speedSq;

                        // Evaporation chance: base chance 0.5% scaled by temp, surface height, and local kinetic energy
                        float base_chance = 0.005f;
                        float evaporation_chance = base_chance * temp_factor * (height_factor * 0.8f + 0.2f) * energy_factor;

                        if (rand_float() < evaporation_chance)
                        {
                            p.type = 4;
                            p.velocity.y += 1.5f; // Evaporated steam rises
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

                        for (size_t j = 0; j < count; ++j)
                        {
                            if (j == i) continue;
                            int o_type = _particles[j].type;
                            if (o_type == 3) // liquid H2O
                            {
                                glm::vec3 diff = {p.position[0] - _particles[j].position[0], p.position[1] - _particles[j].position[1], p.position[2] - _particles[j].position[2]};
                                float distSq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                                if (distSq < search_radius * search_radius)
                                {
                                    near_liquid = true;
                                }
                            }
                            else if (o_type == 4) // vapor H2O
                            {
                                glm::vec3 diff = {p.position[0] - _particles[j].position[0], p.position[1] - _particles[j].position[1], p.position[2] - _particles[j].position[2]};
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

                        if (rand_float() < condensation_chance)
                        {
                            p.type = 3;
                            p.velocity *= 0.5f; // Particle loses energy when condensing to join nearby droplets
                        }
                    }
                }
            }

            // 1. Chemical reaction check (O2 searches for two H2)
            for (size_t i = 0; i < count; ++i)
            {
                Particle& p = _particles[i];
                if (p.type != 2) continue; // TYPE_O2

                float rxn_radius = 0.08f; // increased from 0.06f
                int idx1 = -1;
                int idx2 = -1;

                for (size_t j = 0; j < count; ++j)
                {
                    if (i == j) continue;
                    const Particle& other = _particles[j];
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
                                break;
                            }
                        }
                    }
                }

                if (idx1 != -1 && idx2 != -1)
                {
                    glm::vec3 v_rel1 = _particles[idx1].velocity - p.velocity;
                    glm::vec3 v_rel2 = _particles[idx2].velocity - p.velocity;
                    float E_rel = 0.941f * (glm::dot(v_rel1, v_rel1) + glm::dot(v_rel2, v_rel2));

                    float E_a_kinetic = 1.0f; // lowered from 3.0f
                    if (E_rel > E_a_kinetic)
                    {
                        float E_a_arrhenius = 550.0f; // lowered from 800.0f
                        float P = std::exp(-E_a_arrhenius / sim_params.temp);
                        float r = rand_float();

                        if (r < P)
                        {
                            // Claim both H2 molecules
                            if (_particles[idx1].type == 1 && _particles[idx2].type == 1)
                            {
                                // React!
                                glm::vec3 v_cm = (32.0f * p.velocity + 2.0f * _particles[idx1].velocity + 2.0f * _particles[idx2].velocity) / 36.0f;

                                // Generate direction for emission
                                float theta = rand_float() * 2.0f * 3.14159265f;
                                float z_val = rand_float() * 2.0f - 1.0f;
                                float r_xy = std::sqrt(1.0f - z_val * z_val);
                                glm::vec3 u = {r_xy * std::cos(theta), r_xy * std::sin(theta), z_val};

                                float v_boost = 0.5f;
                                p.velocity = v_cm + v_boost * u;
                                _particles[idx1].velocity = v_cm - v_boost * u;

                                // Boost velocity of nearby molecules (shockwave / heat release)
                                float heat_radius = 0.25f;
                                float heat_boost_strength = 4.0f;
                                for (size_t k = 0; k < count; ++k)
                                {
                                    if (k == i || k == idx1 || k == idx2) continue;
                                    Particle& near_p = _particles[k];
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

                                p.type = 3;
                                p.color = {3.0f, 3.0f, 1.0f}; // Bright yellow/white flash

                                _particles[idx1].type = 3;
                                _particles[idx1].color = {3.0f, 3.0f, 1.0f}; // Bright yellow/white flash

                                _particles[idx2].type = 0;
                                _particles[idx2].color = {0.0f, 0.0f, 0.0f};

                                _host_reactions++;
                            }
                        }
                    }
                }
            }

            // 2. Elastic sphere collisions between all active molecules
            for (size_t i = 0; i < count; ++i)
            {
                Particle& p = _particles[i];
                if (p.type == 0) continue;

                float r_i = get_radius(p.type);
                float m_i = get_mass(p.type);

                for (size_t j = 0; j < count; ++j)
                {
                    if (i == j) continue;
                    Particle& other = _particles[j];
                    if (other.type == 0) continue;

                    float r_j = get_radius(other.type);
                    float m_j = get_mass(other.type);
                    float R_sum = r_i + r_j;

                    glm::vec3 diff = {p.position[0] - other.position[0], p.position[1] - other.position[1], p.position[2] - other.position[2]};
                    float distSq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                    if (distSq < R_sum * R_sum && distSq > 0.000001f)
                    {
                        float dist = std::sqrt(distSq);
                        glm::vec3 n = diff / dist;

                        glm::vec3 v_rel = p.velocity - other.velocity;
                        float v_normal = glm::dot(v_rel, n);

                        if (v_normal < 0.0f)
                        {
                            float e = 0.95f;
                            if (p.type == 3 || other.type == 3) {
                                e = 0.05f; // liquid water collisions are highly inelastic
                            }
                            float J = -(1.0f + e) * v_normal / (1.0f / m_i + 1.0f / m_j);
                            p.velocity += (J / m_i) * n;

                            if (i < j)
                            {
                                _host_collisions++;
                            }
                        }

                        float penetration = R_sum - dist;
                        p.position[0] += 0.5f * penetration * n.x;
                        p.position[1] += 0.5f * penetration * n.y;
                        p.position[2] += 0.5f * penetration * n.z;
                    }
                }
            }

            // 3. Water condensation attraction (cohesion - only for liquid water)
            {
                float cohesion_radius = 0.25f;
                float cohesion_strength = 0.8f; // Gentle attraction strength so it pools slowly
                for (size_t i = 0; i < count; ++i)
                {
                    Particle& p = _particles[i];
                    if (p.type != 3) continue; // Only liquid H2O

                    for (size_t j = 0; j < count; ++j)
                    {
                        if (i == j) continue;
                        const Particle& other = _particles[j];
                        if (other.type != 3) continue;

                        glm::vec3 diff = {p.position[0] - other.position[0], p.position[1] - other.position[1], p.position[2] - other.position[2]};
                        float distSq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                        if (distSq < cohesion_radius * cohesion_radius && distSq > 0.0001f)
                        {
                            float dist = std::sqrt(distSq);
                            glm::vec3 force = -(diff / dist) * cohesion_strength * (1.0f - dist / cohesion_radius);
                            p.velocity += force * dt;
                        }
                    }
                }
            }

            // 4. Update physics
            for (size_t i = 0; i < count; ++i)
            {
                Particle& p = _particles[i];
                if (p.type == 0) // TYPE_INACTIVE
                {
                    p.position = nitros::utils::vec3f{9999.0f, 9999.0f, 9999.0f};
                    p.velocity = glm::vec3(0.0f);
                    out_positions[i] = p.position;
                    out_colors[i] = nitros::utils::vec3f{0.0f, 0.0f, 0.0f};
                    continue;
                }

                float local_temp = sim_params.temp * (1.0f - 0.15f * (p.position[1] / sim_params.boundary));
                if (local_temp < 10.0f) local_temp = 10.0f;

                float gravity = 0.0f;
                float damping = sim_params.damping;
                float bounce = sim_params.bounce;
                bool has_thermal = false;

                if (p.type == 1 || p.type == 2)
                {
                    gravity = 0.0f;
                    damping = 0.999f;
                    bounce = 1.0f;
                    has_thermal = true;
                }
                else if (p.type == 3) // Liquid H2O
                {
                    gravity = sim_params.gravity * 2.0f;
                    damping = 0.90f;
                    bounce = 0.0f;
                    has_thermal = false;

                    // If liquid water is at the bottom, stop its movement completely
                    float limit = sim_params.boundary - sim_params.epsilon;
                    if (p.position[1] <= -limit + 0.08f)
                    {
                        damping = 0.05f;
                        p.velocity.x = 0.0f;
                        p.velocity.z = 0.0f;
                        if (p.velocity.y < 0.0f) p.velocity.y = 0.0f;
                    }
                }
                else if (p.type == 4) // Vapor H2O
                {
                    gravity = 0.0f;
                    damping = 0.999f;
                    bounce = 1.0f;
                    has_thermal = true;

                    // Apply buoyancy: hot steam rises naturally
                    float buoyancy = 0.8f * (local_temp / 300.0f);
                    p.velocity.y += buoyancy * dt;
                }

                p.velocity.y += gravity * dt;

                if (has_thermal)
                {
                    float rx = rand_float() * 2.0f - 1.0f;
                    float ry = rand_float() * 2.0f - 1.0f;
                    float rz = rand_float() * 2.0f - 1.0f;

                    // Thermal speed scales with square root of local temperature
                    float thermal_force = 1.0f * std::sqrt(local_temp / 300.0f);
                    p.velocity.x += rx * thermal_force * dt;
                    p.velocity.y += ry * thermal_force * dt;
                    p.velocity.z += rz * thermal_force * dt;
                }

                p.position[0] += p.velocity.x * dt;
                p.position[1] += p.velocity.y * dt;
                p.position[2] += p.velocity.z * dt;

                // Boundary collisions
                float limit = sim_params.boundary - sim_params.epsilon;
                if (p.position[0] > limit) { p.position[0] = limit; if (p.velocity.x > 0) p.velocity.x *= -bounce; }
                if (p.position[0] < -limit) { p.position[0] = -limit; if (p.velocity.x < 0) p.velocity.x *= -bounce; }
                if (p.position[1] > limit) { p.position[1] = limit; if (p.velocity.y > 0) p.velocity.y *= -bounce; }
                if (p.position[1] < -limit) { p.position[1] = -limit; if (p.velocity.y < 0) p.velocity.y *= -bounce; }
                if (p.position[2] > limit) { p.position[2] = limit; if (p.velocity.z > 0) p.velocity.z *= -bounce; }
                if (p.position[2] < -limit) { p.position[2] = -limit; if (p.velocity.z < 0) p.velocity.z *= -bounce; }

                p.velocity *= damping;

                out_positions[i] = p.position;
                out_colors[i] = p.color;
            }
        }

        void get_particles(std::span<Particle> out_particles) final
        {
            assert(out_particles.size() >= _particles.size());
            std::copy(_particles.begin(), _particles.end(), out_particles.begin());
        }

        unsigned int get_collision_count() const final { return _host_collisions; }
        unsigned int get_reaction_count() const final { return _host_reactions; }
    };
} // namespace cudatr
