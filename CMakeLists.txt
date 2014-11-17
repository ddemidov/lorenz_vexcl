cmake_minimum_required(VERSION 2.8)
project(chaos)

add_definitions(-DVEXCL_CACHE_KERNELS)
add_definitions(-DFUSION_MAX_VECTOR_SIZE=15)

#----------------------------------------------------------------------------
# Find VexCL
#----------------------------------------------------------------------------
set(VEXCL_ROOT $ENV{VEXCL_ROOT} CACHE STRING "VexCL root")
include_directories( ${VEXCL_ROOT} )

#----------------------------------------------------------------------------
# Find odeint
#----------------------------------------------------------------------------
set(ODEINT_ROOT $ENV{ODEINT_ROOT} CACHE STRING "odeint root")
include_directories( ${ODEINT_ROOT} )

#----------------------------------------------------------------------------
# Find Backend
#----------------------------------------------------------------------------
set(VEXCL_BACKEND "OpenCL" CACHE STRING "Select VexCL backend (OpenCL/CUDA)")
set_property(CACHE VEXCL_BACKEND PROPERTY STRINGS "OpenCL" "CUDA")

if ("${VEXCL_BACKEND}" STREQUAL "OpenCL")
    set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
    find_package(OpenCL REQUIRED)
    include_directories( ${OPENCL_INCLUDE_DIRS} )
    set(BACKEND_LIBS ${OPENCL_LIBRARIES})
    add_definitions(-DVEXCL_BACKEND_OPENCL)
elseif ("${VEXCL_BACKEND}" STREQUAL "CUDA")
    find_package(CUDA REQUIRED)
    include_directories( ${CUDA_INCLUDE_DIRS} )
    set(BACKEND_LIBS ${CUDA_CUDA_LIBRARY})
    add_definitions(-DVEXCL_BACKEND_CUDA)
endif()

#----------------------------------------------------------------------------
# Find Boost
#----------------------------------------------------------------------------
set(BOOST_COMPONENTS
    filesystem
    system
    program_options
    )
find_package(Boost COMPONENTS ${BOOST_COMPONENTS})
include_directories( ${Boost_INCLUDE_DIRS} )

#----------------------------------------------------------------------------
# Enable C++11 support, set compilation flags
#----------------------------------------------------------------------------
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
endif ()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

    option(USE_LIBCPP "Use libc++ with Clang" OFF)
    if (USE_LIBCPP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    endif ()
endif ()

#----------------------------------------------------------------------------
foreach(sweep sm_sweep hg_sweep lorenz_sweep new_sweep)
    add_executable(${sweep} ${sweep}.cpp)
    target_link_libraries(${sweep}
        ${Boost_LIBRARIES}
        ${BACKEND_LIBS}
        hdf5 hdf5_cpp
        )
endforeach()

# Copy python scripts to build directory for convenience
if (NOT ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR})
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/plot_kneading
        ${CMAKE_CURRENT_BINARY_DIR}/plot_kneading
        COPYONLY
        )
endif()
