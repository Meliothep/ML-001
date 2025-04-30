cmake_minimum_required(VERSION 3.2)
project(tests)

add_subdirectory(lib/googletest)

include_directories(${MLLOGIC_HEADERS_DIR})
include_directories(lib/googletest/googletest/include)

set(SOURCE_FILES main.cpp src/tests.cpp)

add_executable(tests ${SOURCE_FILES})
target_link_libraries(tests mllogic gtest)
install(TARGETS tests DESTINATION bin)

