cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(002_simple_kalman)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(002_simple_kalman main.cpp)
target_link_libraries(002_simple_kalman "${TORCH_LIBRARIES}")
set_property(TARGET 002_simple_kalman PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET 002_simple_kalman
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:002_simple_kalman>)
endif (MSVC)