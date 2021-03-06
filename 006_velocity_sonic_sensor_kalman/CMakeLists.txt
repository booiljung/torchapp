cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(005_velocity_sonic_sensor_kalman)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(005_velocity_sonic_sensor_kalman main.cpp)
target_link_libraries(005_velocity_sonic_sensor_kalman "${TORCH_LIBRARIES}")
set_property(TARGET 005_velocity_sonic_sensor_kalman PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(
        TARGET 005_velocity_sonic_sensor_kalman
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:005_velocity_sonic_sensor_kalman>)
endif (MSVC)