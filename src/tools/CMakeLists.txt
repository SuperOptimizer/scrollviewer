add_executable(svmanifest svmanifest_main.cpp)
target_include_directories(svmanifest PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(svmanifest PRIVATE sv_options sv_core sv_zarr)
