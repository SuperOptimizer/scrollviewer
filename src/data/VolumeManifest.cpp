#include "data/VolumeManifest.h"

#include <cstring>

namespace sv::data {

namespace {
template <class T>
bool readPod(std::span<const std::byte>& in, T& out) {
  if (in.size() < sizeof(T)) return false;
  std::memcpy(&out, in.data(), sizeof(T));
  in = in.subspan(sizeof(T));
  return true;
}
}  // namespace

Result<std::shared_ptr<VolumeManifest>> VolumeManifest::parse(
    std::span<const std::byte> bytes) {
  std::span<const std::byte> in = bytes;

  char magic[4];
  if (in.size() < 4) return std::unexpected(Error{"manifest too short"});
  std::memcpy(magic, in.data(), 4);
  in = in.subspan(4);
  if (std::memcmp(magic, "SVMF", 4) != 0)
    return std::unexpected(Error{"bad manifest magic"});

  std::uint32_t version = 0;
  std::uint8_t fill = 0, levelCount = 0;
  std::uint16_t reserved = 0;
  if (!readPod(in, version) || !readPod(in, fill) ||
      !readPod(in, levelCount) || !readPod(in, reserved))
    return std::unexpected(Error{"manifest header truncated"});
  if (version != 1)
    return std::unexpected(Error::fmt("unsupported manifest version {}", version));

  auto m = std::make_shared<VolumeManifest>();
  m->fill_ = fill;
  m->levels_.resize(levelCount);

  for (int i = 0; i < levelCount; ++i) {
    Level& l = m->levels_[i];
    if (!readPod(in, l.grid[0]) || !readPod(in, l.grid[1]) ||
        !readPod(in, l.grid[2]))
      return std::unexpected(Error::fmt("manifest level {} header truncated", i));
    const std::size_t n =
        std::size_t{l.grid[0]} * l.grid[1] * l.grid[2] * 2;
    if (in.size() < n)
      return std::unexpected(Error::fmt("manifest level {} data truncated", i));
    l.data.resize(n);
    std::memcpy(l.data.data(), in.data(), n);
    in = in.subspan(n);
  }

  return m;
}

}  // namespace sv::data
