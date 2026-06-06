#pragma once

#include <cuda_runtime.h>
#include <string_view>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            throw std::runtime_error("Cuda Check Error");                     \
        }                                                                     \
    } while (0)

#define CUDA_CHECK_ERR(call)                                                  \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
        }                                                                     \
    } while (0)


struct CheckTime
{
    public:
    CheckTime(std::string_view  message)
        :msg{message}
        ,_st{std::chrono::high_resolution_clock::now()}
    {}

    ~CheckTime()
    {
        auto end_tr = std::chrono::high_resolution_clock::now();
        std::cout<<"Time Dur "<< msg <<"  "<< std::chrono::duration_cast<std::chrono::microseconds>(end_tr - _st).count() << " us " << std::endl;
    }

    private:
    std::string     msg;
    std::chrono::high_resolution_clock::time_point  _st;
};

auto print_dev_attributes(int device_num) -> void;

class CudaContext
{
    uint32_t    _dev_num;
    
    public:
    CudaContext(std::uint32_t  dev_num)
        :_dev_num{dev_num}
    {}

    ~CudaContext() {
        if(cudaDeviceReset() != cudaSuccess) {
            std::cerr<<"Cuda Device Reset Failed"<<std::endl;
        }
    }
};

inline auto create_cuda_device_context(uint32_t  dev_num) -> std::unique_ptr<CudaContext>
{
    return std::make_unique<CudaContext>(dev_num);
}