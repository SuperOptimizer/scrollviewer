find_package(Catch2 3 CONFIG REQUIRED)
find_package(blosc2 CONFIG REQUIRED)

add_executable(sv_tests
  support/ZarrFixture.h
  support/ZarrFixture.cpp
  support/FakeChunkStore.h
  core_tests.cpp
  zarr_tests.cpp
  store_tests.cpp
  pipeline_tests.cpp
)
target_link_libraries(sv_tests PRIVATE
  sv_options sv_core sv_zarr sv_store sv_data
  Catch2::Catch2WithMain Blosc2::blosc2
)
target_include_directories(sv_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(sv_tests PRIVATE
  SV_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
)

include(Catch)
catch_discover_tests(sv_tests)
