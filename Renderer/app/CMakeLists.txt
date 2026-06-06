set(SRC     rotating_cube.cpp 
            particle_updater/ParticleSystem_3.cpp 
            particle_updater/gpu_/particle_updater_gpu.cpp 
            particle_updater/gpu_/gpu_kernel/particle_kernel.cu
            renderer/window_wrapper.cpp
            renderer/particle_renderer.cpp)

add_executable(Rotating_Cube  ${SRC})
target_include_directories(Rotating_Cube PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(Rotating_Cube PUBLIC glCORE GLFW_Interface CuLib)
target_compile_definitions(Rotating_Cube PUBLIC GLCORE_STATIC NITROS_IMAGE_STATIC NITROS_UTILS_STATIC)