cmake_minimum_required(VERSION 3.2)
project(ML001)

add_subdirectory(mllogic)
set(SOURCE_FILES main.cpp)

add_executable(ML001 ${SOURCE_FILES})
target_link_libraries(ML001 "${TORCH_LIBRARIES}" mllogic)
install(TARGETS ML001 DESTINATION ${INSTALL_BIN_DIR})
