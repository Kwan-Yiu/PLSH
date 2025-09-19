cmake_minimum_required(VERSION 3.15)
project(PLSHProject VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

add_executable(plsh
    main.cpp
    plsh.cpp
)

target_link_libraries(plsh PRIVATE Threads::Threads)
