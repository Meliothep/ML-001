find_package(Torch CONFIG REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h" "*.hpp")

set(SOURCE_FILES ${SOURCES} ${HEADERS} )

add_library(mllogic STATIC ${SOURCE_FILES})
target_link_libraries(mllogic PRIVATE "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Arrow::arrow_static,Arrow::arrow_shared>")
target_link_libraries(mllogic PRIVATE "$<IF:$<BOOL:${ARROW_BUILD_STATIC}>,Parquet::parquet_static,Parquet::parquet_shared>")

target_link_libraries(mllogic PRIVATE "${TORCH_LIBRARIES}")

set_property(TARGET mllogic PROPERTY CXX_STANDARD 17)

install(TARGETS mllogic DESTINATION ${INSTALL_LIB_DIR})
install(FILES ${HEADERS} DESTINATION ${INSTALL_INCLUDE_DIR})

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET mllogic
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:mllogic>)
endif (MSVC)

