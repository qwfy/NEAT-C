cmake_minimum_required(VERSION 2.4)

PROJECT(neat)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

FILE(GLOB neat_sources *.cpp)

set(MUJOCO_HOME /opt/mujoco)
include_directories(${MUJOCO_HOME}/include)
link_directories(${MUJOCO_HOME}/bin)
file(GLOB LIB_MUJOCO ${MUJOCO_HOME}/bin/libmujoco[0-9][0-9][0-9].so)

ADD_EXECUTABLE(neat ${neat_sources})
target_link_libraries(neat ${LIB_MUJOCO})

target_link_libraries(neat GL GLU glut )

#Additional libraries from mujoco package
target_link_libraries(neat libglew.so )
target_link_libraries(neat libglfw.so )