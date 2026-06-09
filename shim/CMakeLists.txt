# Build the alloc8-based tracing allocator, libfragtrace_alloc8.
#
#   cmake -S . -B build [-DALLOC8_TAG=v1.0.0]
#   cmake --build build
#
# Produces build/libfragtrace_alloc8.{so,dylib}. Preload it like any alloc8
# allocator (LD_PRELOAD / DYLD_INSERT_LIBRARIES) -- see README.md. The standalone
# C shim (fragtrace.c) has no CMake; build it with `make` instead.

cmake_minimum_required(VERSION 3.16)
project(fragtrace_alloc8 CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# alloc8 ref to fetch. No release tags are published yet, so default to the
# main branch; pin to a specific commit SHA for reproducible builds, e.g.
#   cmake -S . -B build -DALLOC8_TAG=ef0d019
set(ALLOC8_TAG "main" CACHE STRING "alloc8 git ref (branch/tag/sha) to fetch")

include(FetchContent)
FetchContent_Declare(
  alloc8
  GIT_REPOSITORY https://github.com/emeryberger/alloc8.git
  GIT_TAG        ${ALLOC8_TAG}
)
FetchContent_MakeAvailable(alloc8)

# The tracing allocator: our source + alloc8's platform interpose sources.
# ALLOC8_INTERPOSE_SOURCES is exported (CACHE INTERNAL) by alloc8's CMake.
add_library(fragtrace_alloc8 SHARED
  fragtrace_alloc8.cpp
  ${ALLOC8_INTERPOSE_SOURCES}
)
target_link_libraries(fragtrace_alloc8 PRIVATE alloc8::interpose)

# Match the fragtrace naming the Python `collect` driver looks for:
# libfragtrace_alloc8.so / .dylib (the driver also accepts an explicit --tracer).
set_target_properties(fragtrace_alloc8 PROPERTIES
  OUTPUT_NAME "fragtrace_alloc8"
  PREFIX "lib"
)
if(APPLE)
  set_target_properties(fragtrace_alloc8 PROPERTIES SUFFIX ".dylib")
endif()
