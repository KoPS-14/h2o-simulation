#include "./window_wrapper.hpp"

namespace nitros
{
    void Time2::set_time(const Time2::time_point_value  &time_point) noexcept{
        last_time = current_time;
        current_time = time_point;
    }

    auto  Time2::get_time() noexcept -> Time2::time_point_value {
        return current_time;
    }

    auto   Time2::delta_time() noexcept -> Time2::duration_value {
        return current_time - last_time;
    }

    Time2::time_point_value Time2::last_time{};
    Time2::time_point_value Time2::current_time{};
}

#if !defined(__EMSCRIPTEN__)
void key2_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
  if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
      glfwSetWindowShouldClose(window, GL_TRUE);

}
#endif

Window::Window(std::uint32_t width , std::uint32_t  height )
    :dim{width, height}
    ,camera{}
{
    // auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_st>();
    // nitros::glcore::setup_logger({sink})->set_level(spdlog::level::debug);

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);

    auto [version , str] = nitros::glcore::get_version();

    if(version >= 40500) {
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    }
    else {
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    }
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(width, height, "Window", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    glfwSetKeyCallback(window, key2_callback);
}

auto Window::window_dim() const noexcept -> std::pair<std::uint32_t, std::uint32_t> {
    //return std::make_pair(800u, 600u);
    return dim;
}

Window::~Window()
{
    glfwDestroyWindow(window);
}

void Window::run(const std::function<void()>  &funct, std::size_t count )
{
    auto lastFrame = 0.f;

    while(!glfwWindowShouldClose(window) && count != 0)
    {
        funct();

        auto current_frame = glfwGetTime();
        auto deltaTime = current_frame - lastFrame;
        lastFrame = current_frame;

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            camera.ProcessKeyboard(FORWARD, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            camera.ProcessKeyboard(BACKWARD, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            camera.ProcessKeyboard(LEFT, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            camera.ProcessKeyboard(RIGHT, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
            camera.ProcessKeyboard(UP, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
            camera.ProcessKeyboard(DOWN, deltaTime);

        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
            camera.ProcessLook(Camera_Look::Up, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
            camera.ProcessLook(Camera_Look::Down, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
            camera.ProcessLook(Camera_Look::Left, deltaTime);
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
            camera.ProcessLook(Camera_Look::Right, deltaTime);

        glfwSwapBuffers(window);
        glfwPollEvents();
        nitros::Time2::set_time(nitros::Time2::clock_type::now());

        if(count > 0)
            count--;
    }
}