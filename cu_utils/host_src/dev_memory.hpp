#pragma once

#include "./util_fncs.hpp"
#include <memory>
#include <span>
#include <assert.h>

template <typename T>
struct DevDeleter
{
    public:

    void operator()(T *ptr) const 
    {
        if(ptr != nullptr) {
            CUDA_CHECK( cudaFree(static_cast<void*>(ptr)) );
        }
    }
};

  
template <typename T>
using DevMemory_t =  std::unique_ptr<T, DevDeleter<T> >;

template <typename T>
auto create_dev_memory(std::size_t  count_) -> DevMemory_t<T>
{
    void* ptr = nullptr;
    CUDA_CHECK( cudaMalloc(&ptr, count_ * sizeof(T)) );
    return DevMemory_t<T>{reinterpret_cast<T*>(ptr)};
}

template <typename T>
auto create_dev_memory_async(std::size_t  count_, cudaStream_t  cu_stream) -> DevMemory_t<T>
{
    void* ptr = nullptr;
    CUDA_CHECK( cudaMallocAsync(&ptr, count_ * sizeof(T), cu_stream ) );
    return DevMemory_t<T>{reinterpret_cast<T*>(ptr)};
}

template <typename T>
class GpuMemWrap__
{
    private:
    DevMemory_t<T>  _gpu_ptr;
    std::size_t     _count;

    public:
    GpuMemWrap__()
        :_gpu_ptr{}
        ,_count{}
    {}

    GpuMemWrap__(DevMemory_t<T>  &&gpu_ptr, std::size_t  count_)
        :_gpu_ptr{ std::move(gpu_ptr) }
        ,_count{ count_ }
    {}

    GpuMemWrap__(const GpuMemWrap__  &) = delete;
    GpuMemWrap__(GpuMemWrap__  &&other)
        :_gpu_ptr{std::move(other._gpu_ptr)}
        ,_count{std::move(other._count)}
    {
        other._count = 0;
    }

    ~GpuMemWrap__() = default;

    auto operator=(const GpuMemWrap__  &) const -> GpuMemWrap__& = delete;
    auto operator=(GpuMemWrap__  &&other) -> GpuMemWrap__& {
        _gpu_ptr = std::move(other._gpu_ptr);
        _count   = other._count;
        other._count = 0;
        return *this;
    }

    auto allocate_gpu_memory(std::size_t  count_) -> bool
    {
        auto d_mem_ = create_dev_memory<T>(count_);
        _gpu_ptr = std::move(d_mem_);
        _count = count_;
        return true;
    }

    auto allocate_gpu_memory(std::size_t  count_, cudaStream_t  stream_) -> bool 
    {
        auto d_mem_ = create_dev_memory_async<T>(count_, stream_);
        _gpu_ptr = std::move(d_mem_);
        _count = count_;
        return true;
    }

    // auto allocate_managed_memory(std::size_t  count_) -> bool
    // {
    //     void* ptr = nullptr;
    //     CUDA_CHECK( cudaMallocManaged(&ptr, count_ * sizeof(T)) );
    //     _gpu_ptr = DevMemory_t<T>{reinterpret_cast<T*>(ptr)};
    //     _count = count_;
    //     return true;
    // }

    auto count() const -> std::size_t {
        return _count;
    }

    auto size_bytes() const -> std::size_t {
        return _count * sizeof(T);
    }

    auto gpu_ptr() -> T* {
        return _gpu_ptr.get();
    }

    auto gpu_ptr() const -> T* {
        return _gpu_ptr.get();
    }

    auto copy_from_host_to_device(std::span<T>  host_arr) -> void
    {
        if(host_arr.size() > _count)
        {
            auto res = allocate_gpu_memory(host_arr.size());
            assert(res);
        }

        CUDA_CHECK( cudaMemcpy(_gpu_ptr.get(), host_arr.data(), host_arr.size_bytes(), cudaMemcpyKind::cudaMemcpyHostToDevice) );
        _count = host_arr.size();
        return ;
    }

    auto copy_from_host_to_device(std::span<T>  host_arr, cudaStream_t  c_stream_) -> void
    {
        if(host_arr.size() > _count)
        {
            auto res = allocate_gpu_memory(host_arr.size(), c_stream_);
            assert(res);
        }

        CUDA_CHECK( cudaMemcpyAsync(_gpu_ptr.get(), host_arr.data(), host_arr.size_bytes(), cudaMemcpyKind::cudaMemcpyHostToDevice, c_stream_) );
        _count = host_arr.size();
        return ;
    }

    auto copy_from_device_to_host(std::span<T>  host_arr) const -> void
    {
        CUDA_CHECK( cudaMemcpy(host_arr.data(), _gpu_ptr.get(), std::min( (std::size_t)host_arr.size_bytes(), size_bytes() ) , cudaMemcpyKind::cudaMemcpyDeviceToHost) );
        return ;
    }

    auto copy_from_device_to_host(std::span<T>  host_arr, cudaStream_t  c_stream_) const -> void
    {
        CUDA_CHECK( cudaMemcpyAsync(host_arr.data(), _gpu_ptr.get(), std::min( (std::size_t)host_arr.size_bytes(), size_bytes() ) , cudaMemcpyKind::cudaMemcpyDeviceToHost, c_stream_) );
        return ;
    }

    auto copy_from_device_to_device(GpuMemWrap__<T>  &dst_buffer) const -> void
    {
        // Allocate Memory is Less than Host
        if( dst_buffer.count() < count() ) {
            assert( dst_buffer.allocate_gpu_memory(count()) );
        }

        CUDA_CHECK( cudaMemcpy(dst_buffer.gpu_ptr(), _gpu_ptr.get(), std::min( dst_buffer.size_bytes(), size_bytes() ) , cudaMemcpyKind::cudaMemcpyDeviceToDevice) );
        return ;
    }
};

template <typename T>
using GpuMemWrap_t = std::unique_ptr<GpuMemWrap__<T>>;

template <typename T>
inline auto create_gpu_mem_wrap(std::size_t  count_) -> GpuMemWrap_t<T>
{
    auto gpu_mem_ = std::make_unique<GpuMemWrap__<T>>();
    if(count_ > 0) {
        gpu_mem_->allocate_gpu_memory(count_);
    }
    return std::move(gpu_mem_);
}

template <typename T>
inline auto create_gpu_mem_wrap(std::size_t  count_, cudaStream_t  cu_stream_) -> GpuMemWrap_t<T>
{
    auto gpu_mem_ = std::make_unique<GpuMemWrap__<T>>();
    if(count_ > 0) {
        gpu_mem_->allocate_gpu_memory(count_, cu_stream_);
    }
    return std::move(gpu_mem_);
}

template <typename T>
inline auto create_managed_mem_wrap(std::size_t  count_) -> GpuMemWrap_t<T>
{
    auto gpu_mem_ = std::make_unique<GpuMemWrap__<T>>();
    if(count_ > 0) {
        gpu_mem_->allocate_managed_memory(count_);
    }
    return std::move(gpu_mem_);
}