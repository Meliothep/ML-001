cmake_minimum_required(VERSION 3.25)
project(ML001)

find_package(Torch CONFIG REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)

add_subdirectory(mllogic)

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h" "*.hpp")

set(SOURCE_FILES ${SOURCES} ${HEADERS} )

add_executable(ML001 ${SOURCE_FILES})

target_link_libraries(ML001 PRIVATE "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Arrow::arrow_static,Arrow::arrow_shared>")
target_link_libraries(ML001 PRIVATE "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Parquet::parquet_static,Parquet::parquet_shared>")
target_link_libraries(ML001 PRIVATE "${TORCH_LIBRARIES}" mllogic)

set_property(TARGET ML001 PROPERTY CXX_STANDARD 17)

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ML001
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:ML001>)
endif (MSVC)

