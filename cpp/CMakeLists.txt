cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pytorch-examples)

# LibTorch
find_package(Torch QUIET)
if (NOT TORCH_FOUND)
    if (EXISTS "${CMAKE_SOURCE_DIR}/../pytorch-install/")
        set(Torch_DIR "${CMAKE_SOURCE_DIR}/../pytorch-install/share/cmake/Torch")
        find_package(Torch QUIET)
    endif()
endif()
if (NOT TORCH_FOUND)
    if (EXISTS "${CMAKE_SOURCE_DIR}/../libtorch/")
        set(Torch_DIR "${CMAKE_SOURCE_DIR}/../libtorch/share/cmake/Torch")
        find_package(Torch QUIET)
    endif()
endif()

if (TORCH_FOUND)
    message(STATUS "Torch found: ${TORCH_LIBRARIES}")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    include_directories(${TORCH_INCLUDE_DIRS})

    add_definitions(-DUSE_TORCH)
else()
    message(FATAL_ERROR "Torch NOT found")
endif()

message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")

if (TORCH_FOUND)
    message(STATUS "Torch found at ${TORCH_LIBRARIES}")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    include_directories(${TORCH_INCLUDE_DIRS})
endif()

# for each file matching "example*.cpp" in the current directory
file(GLOB EXAMPLES example*.cpp)
foreach(EXAMPLE ${EXAMPLES})
    # get the filename without the extension
    get_filename_component(EXAMPLE_NAME ${EXAMPLE} NAME_WE)
    # add the executable
    add_executable(${EXAMPLE_NAME} ${EXAMPLE})
    # link the executable to the PyTorch libraries
    if (TORCH_FOUND)
        target_link_libraries(${EXAMPLE_NAME} "${TORCH_LIBRARIES}")
        add_definitions(-DUSE_TORCH)
    endif()
    # set the C++ standard to C++20
    set_property(TARGET ${EXAMPLE_NAME} PROPERTY CXX_STANDARD 20)
endforeach(EXAMPLE)

