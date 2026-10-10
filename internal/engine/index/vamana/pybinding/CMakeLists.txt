cmake_minimum_required(VERSION 3.10)
project(vamanalib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
set(Python_ADDITIONAL_VERSIONS 3.8)
find_package(Python 3.8 EXACT COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Add the pybind11 module
pybind11_add_module(vamanalib vamana_py.cpp)

# Include directories
target_include_directories(vamanalib PRIVATE
    ${pybind11_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Add OpenMP support
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    target_link_libraries(vamanalib PRIVATE OpenMP::OpenMP_CXX)
endif()
