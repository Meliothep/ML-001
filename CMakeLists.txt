cmake_minimum_required(VERSION 3.2)

project(cmake-project-template)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})

set(INSTALL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(INSTALL_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)
set(INSTALL_LIB_DIR ${PROJECT_SOURCE_DIR}/lib)

set(MLLOGIC_HEADERS_DIR ${PROJECT_SOURCE_DIR}/src/mllogic)

include_directories(${INSTALL_INCLUDE_DIR})
include_directories(${MLLOGIC_HEADERS_DIR})

add_subdirectory(src)
add_subdirectory(test)


