#include "zarr/OmeZarrVolume.h"

namespace sv::zarr {

namespace {
Result<std::string> readText(const OmeZarrVolume::MetaReader& read,
                             const std::string& relPath) {
  auto buf = read(relPath);
  if (!buf)
    return std::unexpected(Error::fmt("{}: {}", relPath, buf.error().message));
  return std::string(reinterpret_cast<const char*>(buf->data()), buf->size());
}
}  // namespace

Result<std::shared_ptr<OmeZarrVolume>> OmeZarrVolume::open(
    const MetaReader& read) {
  auto attrsText = readText(read, ".zattrs");
  if (!attrsText) return std::unexpected(attrsText.error());
  auto ms = parseOmeAttrs(*attrsText);
  if (!ms) return std::unexpected(ms.error());

  auto vol = std::make_shared<OmeZarrVolume>();
  vol->levels_.reserve(ms->datasets.size());

  for (const auto& ds : ms->datasets) {
    auto zarrayText = readText(read, ds.path + "/.zarray");
    if (!zarrayText) return std::unexpected(zarrayText.error());
    auto meta = parseZarrayV2(*zarrayText);
    if (!meta)
      return std::unexpected(
          Error::fmt("{}/.zarray: {}", ds.path, meta.error().message));

    if (ds.scale[0] != ds.scale[1] || ds.scale[1] != ds.scale[2])
      return std::unexpected(
          Error::fmt("level '{}' has anisotropic scale", ds.path));

    auto codec = makeCodec(meta->compressor);
    if (!codec)
      return std::unexpected(
          Error::fmt("level '{}': {}", ds.path, codec.error().message));

    vol->levels_.push_back(
        Level{ds.path, *meta, ds.scale[0], std::move(*codec)});
  }

  return vol;
}

std::string OmeZarrVolume::chunkStoreKey(int level, std::uint32_t z,
                                         std::uint32_t y,
                                         std::uint32_t x) const {
  const Level& l = levels_[level];
  return l.path + '/' + l.meta.chunkKey(z, y, x);
}

}  // namespace sv::zarr
