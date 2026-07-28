find_package(nlohmann_json CONFIG REQUIRED)
find_package(blosc2 CONFIG REQUIRED)
find_package(zstd CONFIG REQUIRED)

add_library(sv_zarr STATIC
  Dtype.h
  ArrayMeta.h
  ArrayMeta.cpp
  OmeMultiscale.h
  OmeMultiscale.cpp
  Codec.h
  Codec.cpp
  OmeZarrVolume.h
  OmeZarrVolume.cpp
)
target_include_directories(sv_zarr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(sv_zarr
  PUBLIC sv_core
  PRIVATE sv_options nlohmann_json::nlohmann_json Blosc2::blosc2
          zstd::libzstd
)
