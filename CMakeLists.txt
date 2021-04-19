cmake_minimum_required(VERSION 2.8.12)
project(ResNet34)

include_directories(include)
include(~/dlib-19.22/dlib/cmake)

add_executable(main main.cpp src/utility.cpp src/utility.hpp)
target_link_libraries(main dlib::dlib)

# To Compile:
# mkdir build
# cd build
# cmake ..
# cmake --build . --config Release

