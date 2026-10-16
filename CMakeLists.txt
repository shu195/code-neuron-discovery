cmake_minimum_required(VERSION 3.16)
project(CodeNeuronDiscovery VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
else()
    add_compile_options(-O3 -mavx2 -march=native -Wall -Wextra)
endif()

# Main Scientific Solver CLI
add_executable(math_solver src/math_solver_cli.cpp)

# 100-Problem Benchmark Suite
add_executable(benchmark_100 src/benchmark_100.cpp)
