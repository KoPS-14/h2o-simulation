/*----------------------------------------------------------------------------------------------------
Copyright © 2017-2020 Janakiraman Sankara Narayanan (johny.manasee@gmail.com). All Rights Reserved.
----------------------------------------------------------------------------------------------------*/

#ifndef WINDOW_WRAPPER_HPP
#define WINDOW_WRAPPER_HPP

#include <functional>
#include <chrono>
#include <cstdint>

#include <GLFW/glfw3.h>

#include "glcore/glcore_log.hpp"
#include "glcore/context.hpp"
#include "camera/camera.hpp"

#include <iostream>


namespace nitros
{
    class Time2
    {
        public:
        using clock_type = std::chrono::steady_clock;
        using duration_value   = std::chrono::duration<double, std::milli>;
        using time_point_value = std::chrono::time_point<clock_type, duration_value>;

        static void set_time(const time_point_value  &time_point) noexcept;
        [[nodiscard]] static auto get_time() noexcept -> time_point_value;
        [[nodiscard]] static auto delta_time() noexcept -> duration_value;

        private:
        static time_point_value last_time;
        static time_point_value current_time;
    };
}

class Window
{
    public:
    
    Window(std::uint32_t width = 800, std::uint32_t  height = 600);
    auto window_dim() const noexcept -> std::pair<std::uint32_t, std::uint32_t>;
    ~Window();

    void run(const std::function<void()>  &funct, std::size_t   count = -1);

    GLFWwindow*  window;
    Camera  camera;

    private:
    std::pair<std::uint32_t, std::uint32_t>   dim;
    
    bool mv_front = false; 
    bool mv_back  = false;
    bool mv_left  = false;
    bool mv_right = false;
    bool mv_up   = false;
    bool mv_down = false;

    bool cc_up    = false; 
    bool cc_down  = false;
    bool cc_left  = false;
    bool cc_right = false;
};

#endif