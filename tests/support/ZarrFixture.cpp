#include "support/ZarrFixture.h"

#include <blosc2.h>

#include <cstdio>
#include <format>
#include <fstream>
#include <vector>

namespace sv::test {

namespace fs = std::filesystem;

namespace {

void writeTextFile(const fs::path& p, const std::string& text) {
  fs::create_directories(p.parent_path());
  std::ofstream f(p, std::ios::binary);
  f << text;
}

void writeBinaryFile(const fs::path& p, const void* data, std::size_t size) {
  fs::create_directories(p.parent_path());
  std::ofstream f(p, std::ios::binary);
  f.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
}

std::string zarrayJson(const std::uint64_t shape[3], std::uint32_t chunk,
                       bool uint16, bool blosc, bool nested) {
  return std::format(
      R"({{
  "zarr_format": 2,
  "shape": [{}, {}, {}],
  "chunks": [{}, {}, {}],
  "dtype": "{}",
  "order": "C",
  "fill_value": 0,
  "filters": null,
  "dimension_separator": "{}",
  "compressor": {}
}})",
      shape[0], shape[1], shape[2], chunk, chunk, chunk,
      uint16 ? "<u2" : "|u1", nested ? "/" : ".",
      blosc
          ? R"({"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 0, "blocksize": 0})"
          : "null");
}

}  // namespace

std::uint16_t fixtureVoxel(int level, std::uint64_t z, std::uint64_t y,
                           std::uint64_t x) {
  return static_cast<std::uint16_t>(
      (level * 7919 + z * 131 + y * 17 + x * 3) & 0xffff);
}

void writeZarrFixture(const fs::path& root, const ZarrFixtureOptions& opts) {
  const char* attrs = R"({
  "multiscales": [
    {
      "version": "0.4",
      "axes": [
        {"name": "z", "type": "space"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"}
      ],
      "datasets": [
        {
          "path": "0",
          "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}]
        },
        {
          "path": "1",
          "coordinateTransformations": [{"type": "scale", "scale": [2.0, 2.0, 2.0]}]
        }
      ]
    }
  ]
})";
  writeTextFile(root / ".zattrs", attrs);
  writeTextFile(root / ".zgroup", R"({"zarr_format": 2})");

  const std::size_t elem = opts.uint16 ? 2 : 1;
  const std::uint32_t c = kFixtureChunk;

  for (int level = 0; level < 2; ++level) {
    std::uint64_t shape[3];
    for (int i = 0; i < 3; ++i)
      shape[i] = (kFixtureShape[i] + (level ? 1 : 0)) >> level;

    writeTextFile(root / std::to_string(level) / ".zarray",
                  zarrayJson(shape, c, opts.uint16, opts.blosc,
                             opts.nestedSeparator));

    const std::uint64_t grid[3] = {(shape[0] + c - 1) / c,
                                   (shape[1] + c - 1) / c,
                                   (shape[2] + c - 1) / c};
    std::vector<std::byte> chunkBuf(static_cast<std::size_t>(c) * c * c * elem);

    for (std::uint64_t cz = 0; cz < grid[0]; ++cz)
      for (std::uint64_t cy = 0; cy < grid[1]; ++cy)
        for (std::uint64_t cx = 0; cx < grid[2]; ++cx) {
          if (opts.omitChunk000 && level == 0 && cz == 0 && cy == 0 && cx == 0)
            continue;

          // Fill the chunk. Voxels outside the array shape keep value 0
          // (zarr pads edge chunks to full chunk size).
          std::fill(chunkBuf.begin(), chunkBuf.end(), std::byte{0});
          for (std::uint32_t z = 0; z < c; ++z)
            for (std::uint32_t y = 0; y < c; ++y)
              for (std::uint32_t x = 0; x < c; ++x) {
                const std::uint64_t gz = cz * c + z, gy = cy * c + y,
                                    gx = cx * c + x;
                if (gz >= shape[0] || gy >= shape[1] || gx >= shape[2])
                  continue;
                const std::uint16_t v = fixtureVoxel(level, gz, gy, gx);
                const std::size_t off =
                    ((static_cast<std::size_t>(z) * c + y) * c + x) * elem;
                if (opts.uint16) {
                  chunkBuf[off] = static_cast<std::byte>(v & 0xff);        // LE
                  chunkBuf[off + 1] = static_cast<std::byte>(v >> 8);
                } else {
                  chunkBuf[off] = static_cast<std::byte>(v & 0xff);
                }
              }

          const char sep = opts.nestedSeparator ? '/' : '.';
          std::string key = std::to_string(cz);
          key += sep;
          key += std::to_string(cy);
          key += sep;
          key += std::to_string(cx);
          const fs::path chunkPath = root / std::to_string(level) / key;

          if (opts.blosc) {
            // blosc1_compress writes exactly the frame format numcodecs
            // produces for Vesuvius data.
            std::vector<std::byte> comp(chunkBuf.size() + BLOSC2_MAX_OVERHEAD);
            const int n = blosc1_compress(
                3, BLOSC_NOSHUFFLE, elem, chunkBuf.size(), chunkBuf.data(),
                comp.data(), comp.size());
            if (n <= 0) std::abort();  // fixture generation must not fail
            writeBinaryFile(chunkPath, comp.data(),
                            static_cast<std::size_t>(n));
          } else {
            writeBinaryFile(chunkPath, chunkBuf.data(), chunkBuf.size());
          }
        }
  }
}

}  // namespace sv::test
