cmake_minimum_required(VERSION 3.25.2)
project(MyCudaProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA support
enable_language(CUDA)

find_package(CUDAToolkit  REQUIRED)

# Set CUDA architecture (Optional: Adjust for your GPU)
set(CMAKE_CUDA_ARCHITECTURES 86)  # 52 = Maxwell GPUs (Change to your compute capability)

set(SRCS    cu_utils/host_src/util_fncs.cpp)

# Define the CUDA executable
add_library(CuLib STATIC ${SRCS})

# Set compilation flags
target_compile_options(CuLib PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# Separate compilation (needed for linking multiple CUDA files)
set_target_properties(CuLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Set compute capability (Optional: Adjust for your GPU)
set_property(TARGET CuLib PROPERTY CUDA_ARCHITECTURES 52)

target_include_directories(CuLib  PUBLIC  ${CMAKE_SOURCE_DIR}/cu_utils)
target_link_libraries(CuLib PUBLIC CUDA::cudart CUDA::cublas)

# Link CUDA runtime
set_target_properties(CuLib PROPERTIES
                POSITION_INDEPENDENT_CODE  ON 
                CUDA_STANDARD   17)


# add_subdirectory(Img_Processing)
add_subdirectory(Renderer)