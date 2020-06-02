cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

set(PROJECT_NAME 004_velocity_kalman)
project(${PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

find_package(
    Qt5 COMPONENTS
    Widgets REQUIRED
    PrintSupport REQUIRED
    )
if (${Qt5_FOUND})
    message("Found qt5" ${Qt5_VERSION})
else()
    message("Qt5 not found")
endif()

set(
    CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}"    
    )

add_executable(
    ${PROJECT_NAME}
    main.cpp
    qcustomplot.cpp
    )

target_link_libraries(
    ${PROJECT_NAME} PRIVATE
    "${TORCH_LIBRARIES}"
    Qt5::Widgets
    Qt5::PrintSupport
    pthread
    )

set_property(
    TARGET ${PROJECT_NAME}
    PROPERTY CXX_STANDARD 14
    )

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
        )
endif (MSVC)
