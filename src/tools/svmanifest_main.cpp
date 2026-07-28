// svmanifest: generate a VolumeManifest ("SVMF") sidecar for a local
// OME-zarr volume.
//
//   svmanifest <zarr-root> [--scan]
//
// Default mode records existence only (absent chunk -> (fill,fill), present
// -> (0,255)); --scan additionally decodes every present chunk to record
// true per-brick min/max (enables pre-fetch occupancy culling). Production
// manifests should be emitted by the compression pipeline instead.

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#include "core/ThreadPool.h"
#include "zarr/OmeZarrVolume.h"

namespace fs = std::filesystem;
using namespace sv;

namespace {

Result<ByteBuffer> readFile(const fs::path& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) return std::unexpected(Error{"not found"});
  const auto size = static_cast<std::size_t>(f.tellg());
  f.seekg(0);
  auto buf = ByteBuffer::uninitialized(size);
  f.read(reinterpret_cast<char*>(buf.data()),
         static_cast<std::streamsize>(size));
  return buf;
}

template <class T>
void writePod(std::ofstream& out, const T& v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: svmanifest <zarr-root> [--scan]\n");
    return 2;
  }
  const fs::path root = argv[1];
  const bool scan = argc > 2 && std::string_view(argv[2]) == "--scan";

  auto vol = zarr::OmeZarrVolume::open(
      [&](const std::string& rel) { return readFile(root / rel); });
  if (!vol) {
    std::fprintf(stderr, "open failed: %s\n", vol.error().message.c_str());
    return 1;
  }

  const auto fill =
      static_cast<std::uint8_t>((*vol)->level(0).fillValue);
  const int levels = (*vol)->levelCount();

  std::ofstream out(root / "manifest.svmf", std::ios::binary);
  out.write("SVMF", 4);
  writePod(out, std::uint32_t{1});
  writePod(out, fill);
  writePod(out, static_cast<std::uint8_t>(levels));
  writePod(out, std::uint16_t{0});

  for (int li = 0; li < levels; ++li) {
    const auto& meta = (*vol)->level(li);
    const auto grid = meta.chunkGridDims();
    writePod(out, grid[0]);
    writePod(out, grid[1]);
    writePod(out, grid[2]);

    const std::size_t n = std::size_t{grid[0]} * grid[1] * grid[2];
    std::vector<std::uint8_t> data(n * 2);
    std::atomic<std::size_t> present{0};

    // Parallel scan over chunks of this level.
    {
      ThreadPool pool("scan", std::thread::hardware_concurrency());
      std::atomic<std::uint32_t> nextZ{0};
      std::atomic<int> active{0};
      std::mutex doneMutex;
      std::condition_variable doneCv;

      const auto worker = [&](std::stop_token) {
        for (;;) {
          const std::uint32_t z = nextZ.fetch_add(1);
          if (z >= grid[0]) break;
          for (std::uint32_t y = 0; y < grid[1]; ++y)
            for (std::uint32_t x = 0; x < grid[2]; ++x) {
              const std::size_t i =
                  ((std::size_t{z} * grid[1] + y) * grid[2] + x) * 2;
              const fs::path chunk =
                  root / (*vol)->chunkStoreKey(li, z, y, x);
              std::error_code ec;
              if (!fs::exists(chunk, ec)) {
                data[i] = data[i + 1] = fill;
                continue;
              }
              ++present;
              if (!scan) {
                data[i] = 0;
                data[i + 1] = 255;
                continue;
              }
              auto raw = readFile(chunk);
              std::uint8_t lo = 255, hi = 0;
              bool ok = false;
              if (raw) {
                auto dec = ByteBuffer::uninitialized(meta.chunkBytes());
                if ((*vol)->codec(li).decode(raw->span(), dec.span())) {
                  ok = true;
                  if (meta.dtype == zarr::Dtype::U8) {
                    for (const std::byte b : dec.span()) {
                      const auto v = static_cast<std::uint8_t>(b);
                      lo = std::min(lo, v);
                      hi = std::max(hi, v);
                    }
                  } else {
                    const auto* p16 = reinterpret_cast<const std::uint16_t*>(
                        dec.data());
                    for (std::size_t k = 0; k < meta.chunkVoxels(); ++k) {
                      const auto v = static_cast<std::uint8_t>(p16[k] >> 8);
                      lo = std::min(lo, v);
                      hi = std::max(hi, v);
                    }
                  }
                }
              }
              data[i] = ok ? lo : 0;
              data[i + 1] = ok ? hi : 255;
            }
        }
        std::lock_guard lock(doneMutex);
        --active;
        doneCv.notify_all();
      };

      const unsigned workers = pool.threadCount();
      active = static_cast<int>(workers);
      for (unsigned w = 0; w < workers; ++w) pool.post(worker);
      std::unique_lock lock(doneMutex);
      doneCv.wait(lock, [&] { return active == 0; });
    }

    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size()));
    std::printf("level %d: %zux%zux%zu bricks, %zu present (%.1f%%)\n", li,
                std::size_t{grid[0]}, std::size_t{grid[1]},
                std::size_t{grid[2]}, present.load(),
                n ? 100.0 * double(present.load()) / double(n) : 0.0);
  }

  std::printf("wrote %s\n", (root / "manifest.svmf").string().c_str());
  return 0;
}
