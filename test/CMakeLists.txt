cmake_minimum_required(VERSION 3.25)
project(tests)

add_subdirectory(lib/googletest)

include_directories(${MLLOGIC_HEADERS_DIR})
include_directories(lib/googletest/googletest/include)

set(SOURCE_FILES main.cpp src/tests.cpp)

add_executable(tests ${SOURCE_FILES})
set_property(TARGET tests PROPERTY CXX_STANDARD 17)
target_link_libraries(tests mllogic gtest)
install(TARGETS tests DESTINATION bin)

