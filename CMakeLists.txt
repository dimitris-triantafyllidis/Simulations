cmake_minimum_required(VERSION 3.22)

project (
    Simulations
    VERSION 1.0
    LANGUAGES C CXX
)

set ( CMAKE_CXX_STANDARD           23 )
set ( CMAKE_CXX_STANDARD_REQUIRED  ON )
set ( CMAKE_CXX_EXTENSIONS        OFF )

include(FetchContent)

FetchContent_Declare (
    Array
    GIT_REPOSITORY https://github.com/dimitris-triantafyllidis/Array.git
    GIT_TAG main
)

FetchContent_MakeAvailable(Array)

# KP_Solver

add_executable ( KP_Solver
    ${CMAKE_CURRENT_SOURCE_DIR}/source/KP_Solver/KP_Solver.cpp
)

target_link_libraries ( KP_Solver
    PRIVATE
        Array::array
)

# CNN

add_executable ( CNN
    ${CMAKE_CURRENT_SOURCE_DIR}/source/CNN/cnn_example.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/CNN/cnn_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/source/CNN/cnn_utilities.cpp
)

target_link_libraries ( CNN
    PRIVATE
        Array::array
)

# DFT_Solver

add_executable ( DFT_Solver
    ${CMAKE_CURRENT_SOURCE_DIR}/source/DFT_Solver/DFT_Solver.cpp
)

target_link_libraries ( DFT_Solver
    PRIVATE
        Array::array
)

