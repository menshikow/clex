cmake_minimum_required(VERSION 3.10)

project(clex VERSION 0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# You named your executable 'clex' here:
add_executable(clex src/main.cpp)

if(MSVC)
    # So you must use 'clex' here too:
    target_compile_options(clex PRIVATE /W4)
else()
    target_compile_options(clex PRIVATE -Wall -Wextra -pedantic)
endif()
