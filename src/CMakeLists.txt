cmake_minimum_required(VERSION 3.2)
project(ML001)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_subdirectory(mllogic)

set(SOURCE_FILES main.cpp)

add_executable(ML001 ${SOURCE_FILES})
target_link_libraries(ML001 "${TORCH_LIBRARIES}" mllogic)
install(TARGETS ML001 DESTINATION ${INSTALL_BIN_DIR})
set_property(TARGET ML001 PROPERTY CXX_STANDARD 17)

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ML001
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:ML001>)
endif (MSVC)