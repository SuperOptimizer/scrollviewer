if(NOT SV_HAVE_GUI)
  message(STATUS "scrollviewer app skipped (no GUI stack)")
  return()
endif()

find_package(nlohmann_json CONFIG REQUIRED)
set(CMAKE_AUTOMOC ON)

add_executable(scrollviewer WIN32
  main.cpp
  ViewerWindow.h
  ViewerWindow.cpp
  SurfaceWindow.h
  SurfaceWindow.cpp
)
target_include_directories(scrollviewer PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(scrollviewer PRIVATE
  sv_options sv_core sv_data sv_render nlohmann_json::nlohmann_json
  Qt6::Widgets Qt6::OpenGLWidgets
)
vtk_module_autoinit(TARGETS scrollviewer MODULES ${VTK_LIBRARIES})

# Qt needs its platform plugin (qwindows.dll) at runtime; vcpkg copies DLLs
# next to the exe but not the plugin tree.
file(GENERATE
  OUTPUT "$<TARGET_FILE_DIR:scrollviewer>/qt.conf"
  CONTENT "[Paths]\nPlugins = ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/Qt6/plugins\n")
