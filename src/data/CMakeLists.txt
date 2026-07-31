add_library(sv_data STATIC
  BrickKey.h
  RamCache.h
  ChunkFetchPipeline.h
  ChunkFetchPipeline.cpp
  VolumeManifest.h
  VolumeManifest.cpp
  Snic3D.h
  Snic3D.cpp
  TifXyzSurface.h
  TifXyzSurface.cpp
)
target_include_directories(sv_data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
find_package(TIFF REQUIRED)
target_link_libraries(sv_data
  PUBLIC sv_core sv_zarr sv_store
  PRIVATE sv_options TIFF::TIFF
)
