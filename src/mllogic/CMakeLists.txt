find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h" "*.hpp")

set(SOURCE_FILES ${SOURCES} ${HEADERS} )

add_library(mllogic STATIC ${SOURCE_FILES})
target_link_libraries(mllogic "${TORCH_LIBRARIES}")
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