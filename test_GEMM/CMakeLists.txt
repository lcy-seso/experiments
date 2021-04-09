cmake_minimum_required(VERSION 3.13)

project(test_gemm)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wl,--no-undefined -std=c++11")
set(CMAKE_CXX_LINK_EXECUTABLE
    "${CMAKE_CXX_LINK_EXECUTABLE} -lpthread -lm -ldl -lrt -fopenmp -Wl,--no-as-needed"
)

set(MKL_INSTALL_DIR "/data/yincao/intel/oneapi/mkl/latest")
include_directories(${MKL_INSTALL_DIR}/include)
link_directories(${MKL_INSTALL_DIR}/lib/intel64)

add_executable(test main.cpp)
target_link_libraries(test -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread
                      -lmkl_sequential)
