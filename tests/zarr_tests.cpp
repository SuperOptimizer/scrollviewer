#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>

#include "support/ZarrFixture.h"
#include "zarr/ArrayMeta.h"
#include "zarr/Codec.h"
#include "zarr/OmeMultiscale.h"
#include "zarr/OmeZarrVolume.h"

using namespace sv;
using namespace sv::zarr;
using sv::test::ZarrFixtureOptions;

namespace {

const char* kScroll1Zarray = R"({
  "shape": [14376, 7888, 8096],
  "chunks": [128, 128, 128],
  "dtype": "|u1",
  "fill_value": 0,
  "order": "C",
  "filters": null,
  "dimension_separator": "/",
  "compressor": {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 0, "blocksize": 0},
  "zarr_format": 2
})";

Result<ByteBuffer> readFileFrom(const std::filesystem::path& root,
                                const std::string& rel) {
  std::ifstream f(root / rel, std::ios::binary | std::ios::ate);
  if (!f) return std::unexpected(Error{"not found"});
  const auto size = static_cast<std::size_t>(f.tellg());
  f.seekg(0);
  auto buf = ByteBuffer::uninitialized(size);
  f.read(reinterpret_cast<char*>(buf.data()),
         static_cast<std::streamsize>(size));
  return buf;
}

}  // namespace

TEST_CASE("parseZarrayV2 accepts real Scroll1 metadata") {
  auto m = parseZarrayV2(kScroll1Zarray);
  REQUIRE(m.has_value());
  CHECK(m->shape == std::array<std::uint64_t, 3>{14376, 7888, 8096});
  CHECK(m->chunks == std::array<std::uint32_t, 3>{128, 128, 128});
  CHECK(m->dtype == Dtype::U8);
  CHECK(m->dimensionSeparator == '/');
  CHECK(m->compressor == CompressorId::Blosc);
  CHECK(m->chunkGridDims() == std::array<std::uint32_t, 3>{113, 62, 64});
  CHECK(m->chunkKey(12, 3, 4) == "12/3/4");
}

TEST_CASE("parseZarrayV2 handles compressor null and flat separator") {
  auto m = parseZarrayV2(R"({
    "zarr_format": 2, "shape": [10,10,10], "chunks": [8,8,8],
    "dtype": "|u1", "order": "C", "fill_value": 0, "filters": null,
    "compressor": null
  })");
  REQUIRE(m.has_value());
  CHECK(m->compressor == CompressorId::None);
  CHECK(m->dimensionSeparator == '.');
  CHECK(m->chunkKey(1, 2, 3) == "1.2.3");
}

TEST_CASE("parseZarrayV2 rejects unsupported layouts") {
  // order F
  CHECK_FALSE(parseZarrayV2(R"({"zarr_format":2,"shape":[8,8,8],"chunks":[8,8,8],
    "dtype":"|u1","order":"F","compressor":null})").has_value());
  // filters
  CHECK_FALSE(parseZarrayV2(R"({"zarr_format":2,"shape":[8,8,8],"chunks":[8,8,8],
    "dtype":"|u1","order":"C","filters":[{"id":"delta"}],"compressor":null})").has_value());
  // exotic dtype
  CHECK_FALSE(parseZarrayV2(R"({"zarr_format":2,"shape":[8,8,8],"chunks":[8,8,8],
    "dtype":"<f4","order":"C","compressor":null})").has_value());
  // zarr v3
  CHECK_FALSE(parseZarrayV2(R"({"zarr_format":3,"shape":[8,8,8],"chunks":[8,8,8],
    "dtype":"|u1","order":"C","compressor":null})").has_value());
  // garbage
  CHECK_FALSE(parseZarrayV2("not json").has_value());
}

TEST_CASE("parseOmeAttrs reads multiscale datasets and scales") {
  ZarrFixtureOptions opts;
  const auto root =
      std::filesystem::temp_directory_path() / "sv_test_ome_attrs";
  std::filesystem::remove_all(root);
  test::writeZarrFixture(root, opts);

  auto text = readFileFrom(root, ".zattrs");
  REQUIRE(text.has_value());
  auto ms = parseOmeAttrs(
      std::string(reinterpret_cast<const char*>(text->data()), text->size()));
  REQUIRE(ms.has_value());
  REQUIRE(ms->datasets.size() == 2);
  CHECK(ms->datasets[0].path == "0");
  CHECK(ms->datasets[1].scale[0] == 2.0);
}

TEST_CASE("blosc codec round-trips fixture chunks; raw codec too") {
  for (const bool blosc : {true, false}) {
    ZarrFixtureOptions opts;
    opts.blosc = blosc;
    const auto root = std::filesystem::temp_directory_path() /
                      (blosc ? "sv_test_blosc" : "sv_test_raw");
    std::filesystem::remove_all(root);
    test::writeZarrFixture(root, opts);

    auto vol = OmeZarrVolume::open(
        [&](const std::string& rel) { return readFileFrom(root, rel); });
    REQUIRE(vol.has_value());
    REQUIRE((*vol)->levelCount() == 2);

    const auto& meta = (*vol)->level(0);
    auto compressed = readFileFrom(root, (*vol)->chunkStoreKey(0, 1, 1, 1));
    REQUIRE(compressed.has_value());

    auto decoded = ByteBuffer::uninitialized(meta.chunkBytes());
    auto ok = (*vol)->codec(0).decode(compressed->span(), decoded.span());
    REQUIRE(ok.has_value());

    // Chunk (1,1,1) covers voxels [8..15]^3; verify interior and edge padding.
    const std::uint32_t c = test::kFixtureChunk;
    auto at = [&](std::uint32_t z, std::uint32_t y, std::uint32_t x) {
      return static_cast<std::uint8_t>(decoded.span()[(std::size_t(z) * c + y) * c + x]);
    };
    CHECK(at(0, 0, 0) ==
          (test::fixtureVoxel(0, 8, 8, 8) & 0xff));
    CHECK(at(1, 2, 3) ==
          (test::fixtureVoxel(0, 9, 10, 11) & 0xff));
    // gx = 8+5 = 13 >= shape.x (13): padding, must be 0.
    CHECK(at(0, 0, 5) == 0);
  }
}

TEST_CASE("blosc codec rejects truncated and corrupt frames") {
  ZarrFixtureOptions opts;
  const auto root = std::filesystem::temp_directory_path() / "sv_test_trunc";
  std::filesystem::remove_all(root);
  test::writeZarrFixture(root, opts);

  auto vol = OmeZarrVolume::open(
      [&](const std::string& rel) { return readFileFrom(root, rel); });
  REQUIRE(vol.has_value());
  auto compressed = readFileFrom(root, (*vol)->chunkStoreKey(0, 0, 0, 0));
  REQUIRE(compressed.has_value());

  auto decoded = ByteBuffer::uninitialized((*vol)->level(0).chunkBytes());

  SECTION("truncated") {
    compressed->truncate(compressed->size() / 2);
    CHECK_FALSE(
        (*vol)->codec(0).decode(compressed->span(), decoded.span()).has_value());
  }
  SECTION("too short for header") {
    compressed->truncate(4);
    CHECK_FALSE(
        (*vol)->codec(0).decode(compressed->span(), decoded.span()).has_value());
  }
  SECTION("wrong destination size") {
    auto small = ByteBuffer::uninitialized(16);
    CHECK_FALSE(
        (*vol)->codec(0).decode(compressed->span(), small.span()).has_value());
  }
}

TEST_CASE("uint16 fixture decodes with correct values") {
  ZarrFixtureOptions opts;
  opts.uint16 = true;
  const auto root = std::filesystem::temp_directory_path() / "sv_test_u16";
  std::filesystem::remove_all(root);
  test::writeZarrFixture(root, opts);

  auto vol = OmeZarrVolume::open(
      [&](const std::string& rel) { return readFileFrom(root, rel); });
  REQUIRE(vol.has_value());
  CHECK((*vol)->level(0).dtype == Dtype::U16LE);

  auto compressed = readFileFrom(root, (*vol)->chunkStoreKey(0, 0, 0, 0));
  REQUIRE(compressed.has_value());
  auto decoded = ByteBuffer::uninitialized((*vol)->level(0).chunkBytes());
  REQUIRE((*vol)->codec(0).decode(compressed->span(), decoded.span()).has_value());

  const std::uint32_t c = test::kFixtureChunk;
  const auto* p = reinterpret_cast<const std::uint16_t*>(decoded.data());
  CHECK(p[(std::size_t(2) * c + 3) * c + 4] == test::fixtureVoxel(0, 2, 3, 4));
}
