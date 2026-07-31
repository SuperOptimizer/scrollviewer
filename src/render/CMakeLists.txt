# Renderer: VTK/GL brick-pool volume renderer. Starts as a stub; grows with
# milestone 5 (first pixels).
if(NOT SV_HAVE_GUI)
  add_library(sv_render INTERFACE)
  return()
endif()

add_library(sv_render STATIC
  BrickAssembler.h
  BrickAssembler.cpp
  GpuBrickCache.h
  GpuBrickCache.cpp
  GpuUploader.h
  GpuUploader.cpp
  WorkingSetPlanner.h
  WorkingSetPlanner.cpp
  vtkScrollVolumeMapper.h
  vtkScrollVolumeMapper.cpp
  SurfaceMask.h
  SurfaceMask.cpp
  vtkScrollSurfaceMapper.h
  vtkScrollSurfaceMapper.cpp
)
target_include_directories(sv_render PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(sv_render
  PUBLIC sv_core sv_data ${VTK_LIBRARIES}
  PRIVATE sv_options
)
vtk_module_autoinit(TARGETS sv_render MODULES ${VTK_LIBRARIES})
