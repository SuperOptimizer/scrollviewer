find_package(CURL CONFIG REQUIRED)

add_library(sv_store STATIC
  ChunkStore.h
  ChunkStore.cpp
  LocalStore.h
  LocalStore.cpp
  HttpStore.h
  HttpStore.cpp
  DiskCacheStore.h
  DiskCacheStore.cpp
  GdctTranscoder.h
  GdctTranscoder.cpp
)
target_include_directories(sv_store PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(sv_store
  PUBLIC sv_core
  PRIVATE sv_options CURL::libcurl
)
if(SV_HAVE_GPUDCT)
  target_link_libraries(sv_store PRIVATE gpudct::gpudct)
  target_compile_definitions(sv_store PRIVATE SV_HAVE_GPUDCT)
endif()
