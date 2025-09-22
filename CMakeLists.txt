cmake_minimum_required(VERSION 3.15)
project(PLSHProject LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(Python3 3.10 EXACT REQUIRED COMPONENTS Interpreter Development)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local")
endif()

set(USER_PYTHON_SITE "$ENV{HOME}/.local/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")

add_library(plshlib STATIC plsh.cpp)
target_link_libraries(plshlib PRIVATE Threads::Threads)

add_executable(plsh main.cpp)
target_link_libraries(plsh PRIVATE plshlib Threads::Threads)

pybind11_add_module(plsh_python plsh-wrapper.cpp)
target_link_libraries(plsh_python PRIVATE plshlib Threads::Threads)

install(TARGETS plsh_python
        LIBRARY DESTINATION ${USER_PYTHON_SITE}
        RUNTIME DESTINATION ${USER_PYTHON_SITE}
        ARCHIVE DESTINATION ${USER_PYTHON_SITE}
)

