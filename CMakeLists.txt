cmake_minimum_required(VERSION 3.10)

project(clex VERSION 0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(clex src/main.cpp)

if(MSVC)
    target_compile_options(minbpe PRIVATE /W4)
else()
    target_compile_options(minbpe PRIVATE -Wall -Wextra -pedantic)
endif()
