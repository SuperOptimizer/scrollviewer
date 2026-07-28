add_library(sv_core STATIC
  ByteBuffer.h
  Result.h
  ThreadPool.h
  ThreadPool.cpp
  LruCache.h
  Log.h
  Log.cpp
  Profiling.h
  Profiling.cpp
)
target_include_directories(sv_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(sv_core PUBLIC sv_options)
