cmake_minimum_required(VERSION 3.10)

add_library(math_lib add.cpp)

target_include_directories(math_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(main main.cpp)
target_link_libraries(main math_lib)