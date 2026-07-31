cmake_minimum_required(VERSION 3.30)

project(scrollviewer VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 26)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SCROLLVIEWER_BUILD_TESTS "Build unit tests" ON)

# gpudct: lossy bounded-error DCT codec used to recompress raw chunks in the
# disk cache (a 128^3 brick is exactly one Vesuvius zarr chunk). Optional:
# without it the cache stores fetched bytes verbatim.
set(SV_GPUDCT_DIR "D:/gpudct" CACHE PATH "Path to a gpudct source checkout")
if(EXISTS ${SV_GPUDCT_DIR}/CMakeLists.txt)
  set(GPUDCT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GPUDCT_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(GPUDCT_BUILD_BENCH OFF CACHE BOOL "" FORCE)
  add_subdirectory(${SV_GPUDCT_DIR} gpudct EXCLUDE_FROM_ALL)
  set(SV_HAVE_GPUDCT ON)
  message(STATUS "gpudct cache codec: ${SV_GPUDCT_DIR}")
else()
  set(SV_HAVE_GPUDCT OFF)
  message(STATUS "gpudct not found at ${SV_GPUDCT_DIR} - disk cache stores raw bytes")
endif()

# Shared compile options for all scrollviewer targets.
add_library(sv_options INTERFACE)
if(MSVC)
  target_compile_options(sv_options INTERFACE /W4 /permissive- /utf-8 /EHsc)
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # MSVC-cl-only conformance switches (clang-cl conforms by default).
    target_compile_options(sv_options INTERFACE /Zc:preprocessor /Zc:__cplusplus)
  endif()
  target_compile_definitions(sv_options INTERFACE
    NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS
  )
endif()

# GUI stack found once at top level so all modules see the same result.
find_package(Qt6 COMPONENTS Widgets OpenGLWidgets)
find_package(VTK COMPONENTS
  CommonCore
  CommonDataModel
  RenderingCore
  IOImage
  RenderingOpenGL2
  RenderingVolumeOpenGL2
  GUISupportQt
  InteractionStyle
  InteractionWidgets
)
if(Qt6_FOUND AND VTK_FOUND)
  set(SV_HAVE_GUI ON)
else()
  set(SV_HAVE_GUI OFF)
  message(STATUS "Qt6/VTK not found - building data layer only")
endif()

add_subdirectory(src/core)
add_subdirectory(src/zarr)
add_subdirectory(src/store)
add_subdirectory(src/data)
add_subdirectory(src/render)
add_subdirectory(src/app)
add_subdirectory(src/tools)

if(SCROLLVIEWER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
