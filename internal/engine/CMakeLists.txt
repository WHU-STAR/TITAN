cmake_minimum_required(VERSION 3.10)
project(c_api_example)

# Add o3 flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

# Add SIMD optimization flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    # x86_64 SIMD flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2 -mfma")
    message(STATUS "Enabling AVX2 and FMA SIMD instructions for x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    # ARM64 NEON is enabled by default, no additional flags needed
    if(NOT APPLE)
        # Only add -mfpu=neon for non-Apple ARM platforms (like Linux ARM)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpu=neon")
    endif()

    message(STATUS "NEON SIMD instructions available for ARM64 (enabled by default)")
endif()

# Enable fast math optimizations (be careful with precision requirements)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math")

# Set build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# OpenMP
if(APPLE)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    endif()
endif()

find_package(OpenMP)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 设置生成动态库
add_library(hnsw_c_api SHARED
    c_api/hnsw/hnsw_c_api.cpp
)

add_library(ivf_c_api SHARED
    c_api/ivf/ivf_c_api.cpp
)

add_library(vamana_c_api SHARED
    c_api/vamana/vamana_c_api.cpp
)

add_library(example_api SHARED
    c_api/example/example_api.cpp
)

# 添加IVF SIMD基准测试可执行文件
add_executable(ivf_simd_benchmark
    c_api/ivf/simd_benchmark.cpp
)

# 设置头文件包含目录
target_include_directories(hnsw_c_api PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/c_api/hnsw
)

target_include_directories(ivf_c_api PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/c_api/ivf
)

target_include_directories(vamana_c_api PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/c_api/vamana
)

target_include_directories(example_api PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/c_api/example
)

# 设置库的输出名称
set_target_properties(hnsw_c_api PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "hnsw"
)

set_target_properties(ivf_c_api PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "ivf"
)

set_target_properties(vamana_c_api PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "vamana"
)

set_target_properties(example_api PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "example"
)

# 如果是 MacOS，设置 .dylib 后缀
if(APPLE)
    set_target_properties(hnsw_c_api PROPERTIES SUFFIX ".dylib")
    set_target_properties(vamana_c_api PROPERTIES SUFFIX ".dylib")
    set_target_properties(ivf_c_api PROPERTIES SUFFIX ".dylib")
    set_target_properties(example_api PROPERTIES SUFFIX ".dylib")
endif()

# 设置基准测试的头文件包含目录和链接
target_include_directories(ivf_simd_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/c_api/ivf
)
target_link_libraries(ivf_simd_benchmark ivf_c_api)

# 链接 OpenMP
if(OpenMP_CXX_FOUND)
    message("OPENMP FOUND")
    target_link_libraries(vamana_c_api PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(ivf_c_api PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(hnsw_c_api PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(ivf_simd_benchmark OpenMP::OpenMP_CXX)
endif()
