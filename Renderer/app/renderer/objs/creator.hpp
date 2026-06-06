/*----------------------------------------------------------------------------------------------------
Copyright © 2017-2020 Janakiraman Sankara Narayanan (johny.manasee@gmail.com). All Rights Reserved.
----------------------------------------------------------------------------------------------------*/

#ifndef GLCORE_DEV_CREATOR_HPP
#define GLCORE_DEV_CREATOR_HPP

#include "glcore/buffer.hpp"
#include "glcore/textures.h"
// #include "../image_rect.hpp"

#include "checker_gen.hpp"
#include "utilities/memory/memory.hpp"
#include  <tuple>
#include <numeric>
#include <limits>

// #include <generator/PlaneMesh.hpp>
// #include <generator/BoxMesh.hpp>
#if !defined(M_PI)
    #define M_PI 3.14159265358979323846
#endif

namespace nitros
{
    // A struct to hold all attributes for a single vertex
    struct Vertex {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 texCoords;
    };

    // A struct to return all generated mesh data
    struct MeshData_ {
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;
    };

    struct MData
    {
        std::vector<utils::vec3f>   vertices;
        std::vector<utils::vec3f>   normals;
        std::vector<utils::vec2f>   texCoords;
        std::vector<utils::vec3Ui>  indices;
    };

    inline auto convt_(const MeshData_  &src) -> MData {
        auto m_data = MData{};
        for (auto&& val : src.vertices)
        {
            m_data.vertices.push_back( { val.position[0], val.position[1], val.position[2] } );
            m_data.normals.push_back( { val.normal[0], val.normal[1], val.normal[2] } );
            m_data.texCoords.push_back( { val.texCoords[0], val.texCoords[1] } );
        }

        assert(src.indices.size() % 3 == 0);
        for(auto i = 0; i < src.indices.size(); i+=3) {
            m_data.indices.push_back( {
                src.indices.at(i),
                src.indices.at(i+1),
                src.indices.at(i+2),
            } );
        }
        return m_data;
    }


    MeshData_ generateBox(glm::vec3 dimensions, glm::ivec3 subdivisions) {
        MeshData_ mesh{};
        float dx = dimensions.x / subdivisions.x;
        float dy = dimensions.y / subdivisions.y;
        float dz = dimensions.z / subdivisions.z;

        auto createFace = [&](glm::vec3 normal, glm::vec3 tan, glm::vec3 bitan, int subX, int subY, float dimX, float dimY, int startIdx) {
            for (int j = 0; j <= subY; ++j) {
                for (int i = 0; i <= subX; ++i) {
                    glm::vec3 pos = (normal * (dimensions.x/2)) + 
                                    (tan * (static_cast<float>(i) / subX * dimX - dimX/2)) + 
                                    (bitan * (static_cast<float>(j) / subY * dimY - dimY/2));
                    mesh.vertices.push_back({
                        pos,
                        normal,
                        glm::vec2(static_cast<float>(i) / subX, static_cast<float>(j) / subY)
                    });
                }
            }
            for (int j = 0; j < subY; ++j) {
                for (int i = 0; i < subX; ++i) {
                    int rowStart = j * (subX + 1);
                    int nextRowStart = (j + 1) * (subX + 1);
                    mesh.indices.push_back(startIdx + rowStart + i);
                    mesh.indices.push_back(startIdx + nextRowStart + i);
                    mesh.indices.push_back(startIdx + rowStart + i + 1);
                    mesh.indices.push_back(startIdx + nextRowStart + i);
                    mesh.indices.push_back(startIdx + nextRowStart + i + 1);
                    mesh.indices.push_back(startIdx + rowStart + i + 1);
                }
            }
        };
        
        int current_vertex_count = 0;
        
        // Top face (+Y)
        createFace({0,1,0}, {1,0,0}, {0,0,1}, subdivisions.x, subdivisions.z, dimensions.x, dimensions.z, current_vertex_count);
        current_vertex_count = mesh.vertices.size();
        // Bottom face (-Y)
        createFace({0,-1,0}, {1,0,0}, {0,0,-1}, subdivisions.x, subdivisions.z, dimensions.x, dimensions.z, current_vertex_count);
        current_vertex_count = mesh.vertices.size();
        // Front face (+Z)
        createFace({0,0,1}, {1,0,0}, {0,1,0}, subdivisions.x, subdivisions.y, dimensions.x, dimensions.y, current_vertex_count);
        current_vertex_count = mesh.vertices.size();
        // Back face (-Z)
        createFace({0,0,-1}, {-1,0,0}, {0,1,0}, subdivisions.x, subdivisions.y, dimensions.x, dimensions.y, current_vertex_count);
        current_vertex_count = mesh.vertices.size();
        // Right face (+X)
        createFace({1,0,0}, {0,0,-1}, {0,1,0}, subdivisions.z, subdivisions.y, dimensions.z, dimensions.y, current_vertex_count);
        current_vertex_count = mesh.vertices.size();
        // Left face (-X)
        createFace({-1,0,0}, {0,0,1}, {0,1,0}, subdivisions.z, subdivisions.y, dimensions.z, dimensions.y, current_vertex_count);

        return mesh;
    }

    MeshData_ generatePlane(glm::vec2 dimensions, glm::ivec2 subdivisions) 
    {
        MeshData_ mesh;
        float dx = dimensions.x / subdivisions.x;
        float dy = dimensions.y / subdivisions.y;

        for (int j = 0; j <= subdivisions.y; ++j) {
            for (int i = 0; i <= subdivisions.x; ++i) {
                float x = static_cast<float>(i) / subdivisions.x * dimensions.x - dimensions.x / 2.0f;
                float y = static_cast<float>(j) / subdivisions.y * dimensions.y - dimensions.y / 2.0f;
                mesh.vertices.push_back({
                    {x, 0.0f, y},
                    {0.0f, 1.0f, 0.0f},
                    {static_cast<float>(i) / subdivisions.x, static_cast<float>(j) / subdivisions.y}
                });
            }
        }
        
        for (int j = 0; j < subdivisions.y; ++j) {
            for (int i = 0; i < subdivisions.x; ++i) {
                int rowStart = j * (subdivisions.x + 1);
                int nextRowStart = (j + 1) * (subdivisions.x + 1);
                mesh.indices.push_back(rowStart + i);
                mesh.indices.push_back(nextRowStart + i);
                mesh.indices.push_back(rowStart + i + 1);
                mesh.indices.push_back(nextRowStart + i);
                mesh.indices.push_back(nextRowStart + i + 1);
                mesh.indices.push_back(rowStart + i + 1);
            }
        }

        return mesh;
    }

    MeshData_ generateSphere(float radius, int y_divisions, int x_divisions) {
        MeshData_ mesh;
        
        for (int j = 0; j <= y_divisions; ++j) {
            float y_angle = static_cast<float>(j) / y_divisions * M_PI;
            float y_pos = radius * std::cos(y_angle);
            float slice_radius = radius * std::sin(y_angle);
            for (int i = 0; i <= x_divisions; ++i) {
                float x_angle = static_cast<float>(i) / x_divisions * 2.0f * M_PI;
                float x_pos = slice_radius * std::cos(x_angle);
                float z_pos = slice_radius * std::sin(x_angle);
                glm::vec3 position = {x_pos, y_pos, z_pos};
                glm::vec3 normal = glm::normalize(position);
                glm::vec2 texCoords = {static_cast<float>(i) / x_divisions, 1.0f - static_cast<float>(j) / y_divisions};
                mesh.vertices.push_back({position, normal, texCoords});
            }
        }

        for (int j = 0; j < y_divisions; ++j) {
            for (int i = 0; i < x_divisions; ++i) {
                int first_row_start = j * (x_divisions + 1);
                int second_row_start = (j + 1) * (x_divisions + 1);
                mesh.indices.push_back(first_row_start + i);
                mesh.indices.push_back(second_row_start + i);
                mesh.indices.push_back(first_row_start + i + 1);
                mesh.indices.push_back(second_row_start + i);
                mesh.indices.push_back(second_row_start + i + 1);
                mesh.indices.push_back(first_row_start + i + 1);
            }
        }
        
        return mesh;
    }

    MeshData_ generateCone(float radius, float height, int y_divisions, int x_divisions) {
        MeshData_ mesh;
        
        // Top vertex (cone tip)
        mesh.vertices.push_back({ {0, height, 0}, {0,1,0}, {0.5, 1} });
        
        // Cone body
        int body_start_index = mesh.vertices.size();
        for (int j = 0; j <= y_divisions; ++j) {
            float h = height * (1.0f - static_cast<float>(j) / y_divisions);
            float current_radius = radius * (1.0f - static_cast<float>(j) / y_divisions);
            for (int i = 0; i <= x_divisions; ++i) {
                float x_angle = static_cast<float>(i) / x_divisions * 2.0f * M_PI;
                float x = current_radius * std::cos(x_angle);
                float z = current_radius * std::sin(x_angle);
                glm::vec3 pos = {x, h, z};
                glm::vec3 normal = glm::normalize(glm::vec3(x, radius/height * current_radius, z));
                glm::vec2 texCoords = {static_cast<float>(i) / x_divisions, static_cast<float>(j) / y_divisions};
                mesh.vertices.push_back({pos, normal, texCoords});
            }
        }
        
        // Indices for cone body
        for (int j = 0; j < y_divisions; ++j) {
            for (int i = 0; i < x_divisions; ++i) {
                int first_row_start = body_start_index + j * (x_divisions + 1);
                int second_row_start = body_start_index + (j + 1) * (x_divisions + 1);
                mesh.indices.push_back(first_row_start + i);
                mesh.indices.push_back(second_row_start + i);
                mesh.indices.push_back(first_row_start + i + 1);
                mesh.indices.push_back(second_row_start + i);
                mesh.indices.push_back(second_row_start + i + 1);
                mesh.indices.push_back(first_row_start + i + 1);
            }
        }
        
        // Cone base
        int base_center_index = mesh.vertices.size();
        mesh.vertices.push_back({ {0, 0, 0}, {0,-1,0}, {0.5, 0.5} });
        for (int i = 0; i <= x_divisions; ++i) {
            float x_angle = static_cast<float>(i) / x_divisions * 2.0f * M_PI;
            float x = radius * std::cos(x_angle);
            float z = radius * std::sin(x_angle);
            mesh.vertices.push_back({
                {x, 0, z},
                {0, -1, 0},
                {0.5f + std::cos(x_angle)/2.0f, 0.5f + std::sin(x_angle)/2.0f}
            });
        }

        // Indices for cone base
        for (int i = 0; i < x_divisions; ++i) {
            mesh.indices.push_back(base_center_index);
            mesh.indices.push_back(base_center_index + i + 1);
            mesh.indices.push_back(base_center_index + i + 2);
        }
        
        return mesh;
    }


    struct IndexedBuffers
    {
        public:
        utils::Uptr<glcore::Buffer>  vertices;
        utils::Uptr<glcore::Buffer>  normals;
        utils::Uptr<glcore::Buffer>  indices;
        utils::Uptr<glcore::Buffer>  colors;
        utils::Uptr<glcore::Buffer>  tex_coords;

        std::vector<utils::vec3f>   vertices_data;
        std::vector<utils::vec3f>   normals_data;
        std::vector<utils::vec3Ui>  indices_data;
        std::vector<utils::vec3f>   colors_data;
        std::vector<utils::vec2f>   tex_coords_data;
    };

    class Creator
    {
        public:
        Creator(const Creator &) = delete;
        Creator(Creator &&) = delete;
        ~Creator() = default;

        static auto get_instance() -> Creator{
            return Creator{};
        }

        auto indexed_plane() const -> IndexedBuffers {
            auto vert = std::make_unique<glcore::Buffer>();
            auto indi = std::make_unique<glcore::Buffer>();
            auto cols = std::make_unique<glcore::Buffer>();
            auto norm = std::make_unique<glcore::Buffer>();

            const auto cube_pts = cube_points();
            {
                auto& pts = cube_pts;
                auto points = std::decay_t<decltype(pts)>{};
                points.insert(points.begin(), pts.begin(), pts.begin() + 4);
                vert->write_data(points);
            }
            const auto cube_color = cube_colors();
            {
                auto& pts = cube_color;
                auto points = std::decay_t<decltype(pts)>{};
                points.insert(points.begin(), pts.begin(), pts.begin() + 4);
                cols->write_data(points);
            }
            const auto cube_normal = cube_normals();
            {
                auto& pts = cube_normal;
                auto points = std::decay_t<decltype(pts)>{};
                points.insert(points.begin(), pts.begin(), pts.begin() + 4);
                norm->write_data(points);
            }
            const auto cube_ind = cube_index();
            {
                auto& pts = cube_ind;
                auto points = std::decay_t<decltype(pts)>{};
                points.insert(points.begin(), pts.begin(), pts.begin() + 2);
                indi->write_data(points);
            }

            return {
                std::move(vert),
                std::move(norm),
                std::move(indi),
                std::move(cols)
            };
        }

        auto create_index_buffers_from_mesh(const MData &m_data) const -> IndexedBuffers
        {
            auto index_buffer = IndexedBuffers{};

            index_buffer.vertices = std::make_unique<glcore::Buffer>();
            index_buffer.vertices->write_data(m_data.vertices);
            index_buffer.vertices_data = std::move(m_data.vertices);

            index_buffer.normals  = std::make_unique<glcore::Buffer>();
            index_buffer.normals->write_data(m_data.normals);
            index_buffer.normals_data = std::move(m_data.normals);

            index_buffer.tex_coords = std::make_unique<glcore::Buffer>();
            index_buffer.tex_coords->write_data(m_data.texCoords);
            index_buffer.tex_coords_data = std::move(m_data.texCoords);

            index_buffer.indices  = std::make_unique<glcore::Buffer>();
            index_buffer.indices->write_data(m_data.indices);
            index_buffer.indices_data = std::move(m_data.indices);

            return index_buffer;
        }

        template <typename Mesh>
        auto gen_mesh(Mesh &mesh) const
        {
            auto vertices = mesh.vertices();
            auto indices  = mesh.triangles();

            std::vector<utils::vec3f>  position, normal;
            std::vector<utils::vec2f> tex_coord;
            std::vector<utils::vec3Ui> index;

            while (!vertices.done())
            {
                auto v = vertices.generate();
                auto pos = v.position;
                auto nor = v.normal;
                auto tex = v.texCoord;

                position.push_back(
                    {gsl::narrow_cast<float>(pos[0]),
                     gsl::narrow_cast<float>(pos[1]),
                     gsl::narrow_cast<float>(pos[2])
                    });

                normal.push_back(
                    {gsl::narrow_cast<float>(nor[0]),
                     gsl::narrow_cast<float>(nor[1]),
                     gsl::narrow_cast<float>(nor[2])
                    });

                tex_coord.push_back(
                    {gsl::narrow_cast<float>(tex[0]),
                     gsl::narrow_cast<float>(tex[1])
                    });
                vertices.next();
            }

            while (!indices.done())
            {
                auto ind = indices.generate().vertices;

                index.push_back(
                    {
                        gsl::narrow_cast<std::uint32_t>(ind[0]),
                        gsl::narrow_cast<std::uint32_t>(ind[1]),
                        gsl::narrow_cast<std::uint32_t>(ind[2])
                    }
                );

                indices.next();
            }

            auto index_buffer = IndexedBuffers{};

            index_buffer.vertices = std::make_unique<glcore::Buffer>();
            index_buffer.vertices->write_data(position);
            index_buffer.vertices_data = std::move(position);

            index_buffer.normals  = std::make_unique<glcore::Buffer>();
            index_buffer.normals->write_data(normal);
            index_buffer.normals_data = std::move(normal);

            index_buffer.indices  = std::make_unique<glcore::Buffer>();
            index_buffer.indices->write_data(index);
            index_buffer.indices_data = std::move(index);

            index_buffer.tex_coords = std::make_unique<glcore::Buffer>();
            index_buffer.tex_coords->write_data(tex_coord);
            index_buffer.tex_coords_data = std::move(tex_coord);

            return index_buffer;
        }

        auto indexed_plane(const std::array<double,2>  &dim, const utils::vec2i  &seg) const
        {
            // auto plane = generator::PlaneMesh{{dim[0], dim[1]}, {seg[0], seg[1]}};
            auto plane = generatePlane( {dim[0], dim[1]}, {seg[0], seg[1]} );
            std::cout<<"Plane Generated "<<std::endl;
            return create_index_buffers_from_mesh( convt_(plane) );
        }

        auto indexed_cube2(const utils::vec3f &dim, const utils::vec3i  &segs) const -> IndexedBuffers 
        {
            // auto cube = generator::BoxMesh{{dim[0], dim[1], dim[2]}, {segs[0], segs[1], segs[2]}};
            auto cube = generateBox({dim[0], dim[1], dim[2]}, {segs[0], segs[1], segs[2]});
            std::cout<<"Box Generated "<<std::endl;
            auto c_cube = convt_(cube);
            std::cout<<"Converted Box Generated "<<std::endl;
            auto id_buffer = create_index_buffers_from_mesh( c_cube );
            std::cout<<"Index Buffers From Mesh "<<std::endl;
            return id_buffer;
        }

        // Gives color
        auto indexed_cube() const -> IndexedBuffers 
        {
            auto vert = std::make_unique<glcore::Buffer>();
            auto indi = std::make_unique<glcore::Buffer>();
            auto cols = std::make_unique<glcore::Buffer>();
            auto norm = std::make_unique<glcore::Buffer>();

            vert->write_data(cube_points());
            indi->write_data(cube_index());
            cols->write_data(cube_colors());
            norm->write_data(cube_normals());

            return {
                std::move(vert),
                std::move(norm),
                std::move(indi),
                std::move(cols)
            };
        }   

        private:

        auto cube_points(const utils::vec3f  &offset = {0.f, 0.f, 0.f}, const utils::vec3f  &dim = {0.5f, 0.5f, 0.5f}) const -> std::vector<utils::vec3f>{
            return {
                // front
             {-dim[0] + offset[0], -dim[1] + offset[1],  dim[2] + offset[2]},
             { dim[0] + offset[0], -dim[1] + offset[1],  dim[2] + offset[2]},
             { dim[0] + offset[0],  dim[1] + offset[1],  dim[2] + offset[2]},
             {-dim[0] + offset[0],  dim[1] + offset[1],  dim[2] + offset[2]},
             // back
             {-dim[0] + offset[0], -dim[1] + offset[1], -dim[2] + offset[2]},
             { dim[0] + offset[0], -dim[1] + offset[1], -dim[2] + offset[2]},
             { dim[0] + offset[0],  dim[1] + offset[1], -dim[2] + offset[2]},
             {-dim[0] + offset[0],  dim[1] + offset[1], -dim[2] + offset[2]}
            };
        }

        auto cube_index() const -> std::vector<utils::vec3Ui> {
            return {
                {0, 1, 2},
                {2, 3, 0},

                {4, 5, 6},
                {6, 7, 4}
            };
        }

        auto cube_colors() const -> std::vector<utils::vec3f> {
            return {
             {1.0, 0.0, 0.0},
             {0.0, 1.0, 0.0},
             {0.0, 0.0, 1.0},
             {1.0, 1.0, 1.0},

             {1.0, 0.0, 0.0},
             {0.0, 1.0, 0.0},
             {0.0, 0.0, 1.0},
             {1.0, 1.0, 1.0}
            };
        }

        auto cube_normals() const -> std::vector<utils::vec3f> {
            return {
             {0.0, 0.0, 1.0},
             {0.0, 0.0, 1.0},
             {0.0, 0.0, 1.0},
             {0.0, 0.0, 1.0},

             {0.0, 0.0,-1.0},
             {0.0, 0.0,-1.0},
             {0.0, 0.0,-1.0},
             {0.0, 0.0,-1.0},
            };
        }


        Creator() = default;
    };

}
#endif