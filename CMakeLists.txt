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
