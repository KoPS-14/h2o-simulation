project(Renderer)

add_library(GSL_lib INTERFACE)
target_include_directories(GSL_lib  INTERFACE  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/dep_/gsl/include>
                                            $<INSTALL_INTERFACE:include>)                          

add_library(GLM_lib INTERFACE)
target_include_directories(GLM_lib  INTERFACE  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/dep_/glm/include>
                                            $<INSTALL_INTERFACE:include>)                          

# Turn off GLFW examples in the cache
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build the GLFW example programs" FORCE)

add_subdirectory(dep_/GLFW/glfw-3.3.2)
add_library(GLFW_Interface INTERFACE)
target_include_directories(GLFW_Interface INTERFACE ${CMAKE_CURRENT_LIST_DIR}/dep_/GLFW/glfw-3.3.2/include)
target_link_libraries(GLFW_Interface INTERFACE glfw)

add_subdirectory(dep_/utilities)
add_subdirectory(dep_/images)
add_subdirectory(dep_/GLCore)


add_subdirectory(app)