#include "./util_fncs.hpp"
#include "device_launch_parameters.h"

auto print_dev_attributes(int device_num) -> void
{
    {
        int value = {};
        CUDA_CHECK_ERR( cudaDeviceGetAttribute(&value, cudaDeviceAttr::cudaDevAttrComputeCapabilityMajor, device_num) );
        std::cout<<"Major Version "<<value<<std::endl;

        CUDA_CHECK_ERR( cudaDeviceGetAttribute(&value, cudaDeviceAttr::cudaDevAttrComputeCapabilityMinor, device_num) );
        std::cout<<"Minor Version "<<value<<std::endl;

        CUDA_CHECK_ERR( cudaDeviceGetAttribute(&value, cudaDeviceAttr::cudaDevAttrMultiProcessorCount, device_num) );
        std::cout<<"Streaming Multi Processor Count "<<value<<std::endl;

        CUDA_CHECK_ERR( cudaDeviceGetAttribute(&value, cudaDeviceAttr::cudaDevAttrMaxThreadsPerMultiProcessor, device_num) );
        std::cout<<"Max Threads Per SM "<<value<<std::endl;

        std::cout<<"\n\n"<<std::endl;

        CUDA_CHECK_ERR( cudaDeviceGetAttribute(&value, cudaDeviceAttr::cudaDevAttrMaxThreadsPerBlock, device_num) );
        std::cout<<"Max Threads per Block "<<value<<std::endl;

        CUDA_CHECK_ERR( cudaDeviceGetAttribute(&value, cudaDeviceAttr::cudaDevAttrMaxBlockDimX, device_num) );
        std::cout<<"Max Block Dim X "<<value<<std::endl;

        CUDA_CHECK_ERR( cudaDeviceGetAttribute(&value, cudaDeviceAttr::cudaDevAttrMaxBlockDimY, device_num) );
        std::cout<<"Max Block Dim Y "<<value<<std::endl;

        CUDA_CHECK_ERR( cudaDeviceGetAttribute(&value, cudaDeviceAttr::cudaDevAttrMaxBlockDimZ, device_num) );
        std::cout<<"Max Block Dim Z "<<value<<std::endl;

        
    }
    // cudaDeviceGetAttribute()
}