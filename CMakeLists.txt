cmake_minimum_required(VERSION 2.4)

PROJECT(neat)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

FILE(GLOB neat_sources *.cpp)

ADD_EXECUTABLE(neat ${neat_sources})
