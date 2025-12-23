cmake_minimum_required(VERSION 3.10)

project(clex VERSION 0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Source files
set(SOURCES
    src/main.cpp
    src/bpe.cpp
    src/io.cpp
    src/cli.cpp
)

# Create executable
add_executable(clex ${SOURCES})

# Set include directories
target_include_directories(clex PRIVATE include)

# Compiler options
if(MSVC)
    target_compile_options(clex PRIVATE /W4)
else()
    target_compile_options(clex PRIVATE -Wall -Wextra -pedantic)
endif()
