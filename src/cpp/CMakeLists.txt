cmake_minimum_required(VERSION 3.14)
# Project metadata
project(PositLib
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS On)
message(STATUS "compiler = " ${CMAKE_C_COMPILER_ID})

include_directories(/usr/include/openblas) # In case cblas is not found
find_package(BLAS REQUIRED)

# set(target, DEBUG)
set(target, RELEASE)

# Set C++ Standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Ensure strict standard compliance (-std=c++20)

# --- Compiler Options ---
# Enable standard warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- Posit Library ---
# Define the library target
add_library(posit_lib STATIC
    posit.cc
    posit.h
	packed_posit.h
)

# Ensure the current directory is included so headers are found
target_include_directories(posit_lib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# --- Posit test ---
add_executable(posit_test 
    posit_test.cc
)
target_link_libraries(posit_test PRIVATE posit_lib)

# --- CPU Posit DGEMV ---
add_executable(cpu_posit_dgemv 
    cpu_posit_gemv.cc
	posit_gemv.h
)
target_link_libraries(cpu_posit_dgemv PRIVATE posit_lib ${BLAS_LIBRARIES})
