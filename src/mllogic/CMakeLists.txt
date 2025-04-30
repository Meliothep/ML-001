cmake_minimum_required(VERSION 3.2)
project(mllogic C CXX)

set(Torch_DIR ${INSTALL_LIB_DIR}/libtorch/share/cmake/Torch)

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h" "*.hpp")

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(SOURCE_FILES ${SOURCES} ${HEADERS} )

add_library(mllogic SHARED STATIC ${SOURCE_FILES})

install(TARGETS mllogic DESTINATION ${INSTALL_LIB_DIR})
install(FILES ${HEADERS} DESTINATION ${INSTALL_INCLUDE_DIR})