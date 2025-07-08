cmake_minimum_required(VERSION 3.25)
cmake_policy(VERSION 3.25)
cmake_policy(SET CMP0074 NEW)  # Fix OpenCV_ROOT warning

project(ML001)

# Force Protobuf to be found first to prevent partial target registration
find_package(Protobuf CONFIG REQUIRED)

# Torch
find_package(Torch CONFIG REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Arrow / Parquet
find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)

add_subdirectory(mllogic)

file(GLOB SOURCES "*.cpp" "models/*.cpp")
file(GLOB HEADERS "*.h" "*.hpp" "models/*.h" "models/*.hpp")
add_executable(ML001 ${SOURCES} ${HEADERS})

add_compile_definitions(NOMINMAX)

target_link_libraries(ML001 PRIVATE
        "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Arrow::arrow_static,Arrow::arrow_shared>"
        "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Parquet::parquet_static,Parquet::parquet_shared>"
        "${TORCH_LIBRARIES}"
        "${OpenCV_LIBS}"
        mllogic
)

set_property(TARGET ML001 PROPERTY CXX_STANDARD 17)

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ML001
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:ML001>
    )
endif()
