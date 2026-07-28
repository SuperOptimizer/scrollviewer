#include "render/GpuBrickCache.h"

#include <vtk_glew.h>

#include <vtkOpenGLRenderWindow.h>

#include <algorithm>
#include <cmath>

#include "core/Log.h"

namespace sv::render {

using data::BrickKey;

namespace {
constexpr std::uint32_t kTileEntries =
    GpuBrickCache::kTileDim * GpuBrickCache::kTileDim * GpuBrickCache::kTileDim;
}

GpuBrickCache::GpuBrickCache(std::vector<LevelDims> levels,
                             std::uint32_t borderedDim,
                             std::uint64_t poolByteBudget)
    : borderedDim_(borderedDim), poolBudget_(poolByteBudget) {
  levels_.resize(levels.size());
  std::size_t totalTilesUpperBound = 0;
  for (std::size_t i = 0; i < levels.size(); ++i) {
    Level& l = levels_[i];
    l.grid = levels[i].grid;
    for (int a = 0; a < 3; ++a)
      l.dirDim[a] = (l.grid[a] + kTileDim - 1) / kTileDim;
    l.dirMirror.assign(
        std::size_t{l.dirDim[0]} * l.dirDim[1] * l.dirDim[2], kDirUnknown);
    totalTilesUpperBound +=
        std::size_t{l.dirDim[0]} * l.dirDim[1] * l.dirDim[2];
  }

  // Tile pool capacity: enough for every tile of every level (upper bound),
  // but only the mirrors are sized eagerly; GPU memory is one fixed texture.
  // A 16^3-entry tile is 16 KB of entries + 8 KB occupancy.
  tileCapacity_ = static_cast<std::uint32_t>(
      std::min<std::size_t>(totalTilesUpperBound, 65536));
  {
    const auto tilesAxis = static_cast<std::uint32_t>(
        std::ceil(std::cbrt(double(tileCapacity_))));
    tilePoolDims_ = {tilesAxis, tilesAxis,
                     (tileCapacity_ + tilesAxis * tilesAxis - 1) /
                         (tilesAxis * tilesAxis)};
    // Recompute capacity to the full box so slot math is uniform.
    tileCapacity_ = tilePoolDims_[0] * tilePoolDims_[1] * tilePoolDims_[2];
  }
  tileMirror_.assign(std::size_t{tileCapacity_} * kTileEntries, 0u);
  occMirror_.assign(std::size_t{tileCapacity_} * kTileEntries * 2, 0);
  tileDirty_.assign(tileCapacity_, false);

  // Brick pool sizing (unchanged from the flat-table design).
  const std::uint64_t slotBytes =
      std::uint64_t{borderedDim_} * borderedDim_ * borderedDim_;
  const std::uint64_t wantSlots =
      std::max<std::uint64_t>(8, poolBudget_ / slotBytes);
  const std::uint32_t maxAxisSlots = 2048 / borderedDim_;
  std::uint32_t s =
      static_cast<std::uint32_t>(std::ceil(std::cbrt(double(wantSlots))));
  s = std::clamp(s, 2u, maxAxisSlots);
  std::uint32_t sz = s, sy = s;
  std::uint32_t sx = static_cast<std::uint32_t>(std::clamp<std::uint64_t>(
      wantSlots / (std::uint64_t{sz} * sy), 1, maxAxisSlots));
  slotDims_ = {sz, sy, sx};

  const std::size_t total = std::size_t{sz} * sy * sx;
  slots_.resize(total);
  freeSlots_.reserve(total);
  for (std::size_t i = total; i > 0; --i)
    freeSlots_.push_back(static_cast<std::uint32_t>(i - 1));

  logInfo(
      "brick pool: {}x{}x{} slots of {}^3 ({} MB); tile pool: {} tiles "
      "({} MB mirrors)",
      sx, sy, sz, borderedDim_, total * slotBytes >> 20, tileCapacity_,
      (tileMirror_.size() * 4 + occMirror_.size()) >> 20);
}

GpuBrickCache::~GpuBrickCache() = default;

void GpuBrickCache::initialize(vtkOpenGLRenderWindow* renWin) {
  renWin_ = renWin;
  logInfo("OpenGL renderer: {} ({})",
          reinterpret_cast<const char*>(glGetString(GL_RENDERER)),
          reinterpret_cast<const char*>(glGetString(GL_VENDOR)));

  const auto makeTex = [&](int wrapNearest) {
    auto t = vtkSmartPointer<vtkTextureObject>::New();
    t->SetContext(renWin);
    t->SetWrapS(vtkTextureObject::ClampToEdge);
    t->SetWrapT(vtkTextureObject::ClampToEdge);
    t->SetWrapR(vtkTextureObject::ClampToEdge);
    t->SetMinificationFilter(wrapNearest ? vtkTextureObject::Nearest
                                         : vtkTextureObject::Linear);
    t->SetMagnificationFilter(wrapNearest ? vtkTextureObject::Nearest
                                          : vtkTextureObject::Linear);
    return t;
  };

  poolHolder_ = makeTex(0);
  poolHolder_->Create3DFromRaw(slotDims_[2] * borderedDim_,
                               slotDims_[1] * borderedDim_,
                               slotDims_[0] * borderedDim_, 1,
                               VTK_UNSIGNED_CHAR, nullptr);
  pool_ = poolHolder_;

  for (auto& l : levels_) {
    l.dirTex = makeTex(1);
    l.dirTex->SetRequireTextureInteger(true);
    l.dirTex->SetInternalFormat(GL_R32UI);
    l.dirTex->SetFormat(GL_RED_INTEGER);
    l.dirTex->SetDataType(GL_UNSIGNED_INT);
    l.dirTex->Create3DFromRaw(l.dirDim[2], l.dirDim[1], l.dirDim[0], 1,
                              VTK_UNSIGNED_INT, l.dirMirror.data());
    l.dirDirty = false;
  }

  tilePoolTex_ = makeTex(1);
  tilePoolTex_->SetRequireTextureInteger(true);
  tilePoolTex_->SetInternalFormat(GL_R32UI);
  tilePoolTex_->SetFormat(GL_RED_INTEGER);
  tilePoolTex_->SetDataType(GL_UNSIGNED_INT);
  tilePoolTex_->Create3DFromRaw(tilePoolDims_[2] * kTileDim,
                                tilePoolDims_[1] * kTileDim,
                                tilePoolDims_[0] * kTileDim, 1,
                                VTK_UNSIGNED_INT, tileMirror_.data());

  occPoolTex_ = makeTex(1);
  occPoolTex_->Create3DFromRaw(tilePoolDims_[2] * kTileDim,
                               tilePoolDims_[1] * kTileDim,
                               tilePoolDims_[0] * kTileDim, 2,
                               VTK_UNSIGNED_CHAR, occMirror_.data());
}

void GpuBrickCache::releaseGraphicsResources() {
  for (auto& l : levels_) {
    if (l.dirTex) l.dirTex->ReleaseGraphicsResources(renWin_);
    l.dirTex = nullptr;
  }
  if (tilePoolTex_) tilePoolTex_->ReleaseGraphicsResources(renWin_);
  if (occPoolTex_) occPoolTex_->ReleaseGraphicsResources(renWin_);
  tilePoolTex_ = nullptr;
  occPoolTex_ = nullptr;
  if (poolHolder_) poolHolder_->ReleaseGraphicsResources(renWin_);
  poolHolder_ = nullptr;
  pool_ = nullptr;
  renWin_ = nullptr;
}

std::uint32_t GpuBrickCache::readEntry(const BrickKey& key) const {
  const auto& c = key.coord;
  const Level& l = levels_[c.level];
  const std::uint32_t d = l.dirMirror[dirIndex(l, c.z, c.y, c.x)];
  if (d == kDirUnknown) return 0u;
  if (d == kDirAllEmpty) return kEmptyBit;
  const std::size_t base = std::size_t{d - 2} * kTileEntries;
  return tileMirror_[base + inTileIndex(c.z, c.y, c.x)];
}

std::uint32_t GpuBrickCache::allocateTile(Level& l, std::size_t dirIdx) {
  if (tilesAllocated_ >= tileCapacity_) {
    logWarn("page-tile pool exhausted ({} tiles)", tileCapacity_);
    return UINT32_MAX;
  }
  const std::uint32_t tile = tilesAllocated_++;
  const std::size_t base = std::size_t{tile} * kTileEntries;
  const std::uint32_t fillEntry =
      (l.dirMirror[dirIdx] == kDirAllEmpty) ? kEmptyBit : 0u;
  std::fill_n(tileMirror_.begin() + base, kTileEntries, fillEntry);
  std::fill_n(occMirror_.begin() + base * 2, kTileEntries * 2, 0);
  l.dirMirror[dirIdx] = tile + 2;
  l.dirDirty = true;
  tileDirty_[tile] = true;
  return tile;
}

void GpuBrickCache::writeEntry(const BrickKey& key, std::uint32_t value,
                               std::uint8_t occMin, std::uint8_t occMax) {
  const auto& c = key.coord;
  Level& l = levels_[c.level];
  const std::size_t di = dirIndex(l, c.z, c.y, c.x);
  std::uint32_t d = l.dirMirror[di];
  if (d < 2) {
    const std::uint32_t tile = allocateTile(l, di);
    if (tile == UINT32_MAX) return;
    d = tile + 2;
  }
  const std::uint32_t tile = d - 2;
  const std::size_t idx =
      std::size_t{tile} * kTileEntries + inTileIndex(c.z, c.y, c.x);
  tileMirror_[idx] = value;
  occMirror_[idx * 2] = occMin;
  occMirror_[idx * 2 + 1] = occMax;
  tileDirty_[tile] = true;
}

bool GpuBrickCache::isResident(const BrickKey& key) const {
  if (slotOf_.contains(key)) return true;
  return (readEntry(key) & kEmptyBit) != 0;
}

void GpuBrickCache::touch(const BrickKey& key) {
  if (auto it = slotOf_.find(key); it != slotOf_.end())
    slots_[it->second].lastTouched = frame_;
}

void GpuBrickCache::markEmpty(const BrickKey& key) {
  if ((readEntry(key) & kEmptyBit) != 0) return;
  writeEntry(key, kEmptyBit, 0, 0);
}

void GpuBrickCache::seedEmpties(
    int level,
    const std::function<bool(std::uint32_t, std::uint32_t, std::uint32_t)>&
        isEmpty) {
  Level& l = levels_[level];
  for (std::uint32_t tz = 0; tz < l.dirDim[0]; ++tz)
    for (std::uint32_t ty = 0; ty < l.dirDim[1]; ++ty)
      for (std::uint32_t tx = 0; tx < l.dirDim[2]; ++tx) {
        bool all = true, any = false;
        const std::uint32_t z1 = std::min(l.grid[0], (tz + 1) * kTileDim);
        const std::uint32_t y1 = std::min(l.grid[1], (ty + 1) * kTileDim);
        const std::uint32_t x1 = std::min(l.grid[2], (tx + 1) * kTileDim);
        for (std::uint32_t z = tz * kTileDim; z < z1 && (all || !any); ++z)
          for (std::uint32_t y = ty * kTileDim; y < y1 && (all || !any); ++y)
            for (std::uint32_t x = tx * kTileDim; x < x1 && (all || !any);
                 ++x) {
              if (isEmpty(z, y, x))
                any = true;
              else
                all = false;
            }
        const std::size_t di =
            (std::size_t{tz} * l.dirDim[1] + ty) * l.dirDim[2] + tx;
        if (all && any && l.dirMirror[di] == kDirUnknown) {
          l.dirMirror[di] = kDirAllEmpty;  // whole tile: one dir entry
          l.dirDirty = true;
        } else if (any) {
          for (std::uint32_t z = tz * kTileDim; z < z1; ++z)
            for (std::uint32_t y = ty * kTileDim; y < y1; ++y)
              for (std::uint32_t x = tx * kTileDim; x < x1; ++x)
                if (isEmpty(z, y, x))
                  markEmpty(BrickKey{
                      0, data::ChunkCoord{static_cast<std::uint8_t>(level), z,
                                          y, x}});
        }
      }
}

std::uint32_t GpuBrickCache::acquireSlot(const BrickKey& key) {
  if (!freeSlots_.empty()) {
    const std::uint32_t s = freeSlots_.back();
    freeSlots_.pop_back();
    return s;
  }
  std::uint32_t victim = UINT32_MAX;
  std::uint64_t oldest = UINT64_MAX;
  for (std::uint32_t i = 0; i < slots_.size(); ++i) {
    const Slot& s = slots_[i];
    if (!s.occupied || s.pinned) continue;
    if (s.lastTouched < oldest) {
      oldest = s.lastTouched;
      victim = i;
    }
  }
  if (victim == UINT32_MAX) {
    logWarn("brick pool exhausted by pinned bricks; dropping upload");
    return UINT32_MAX;
  }
  writeEntry(slots_[victim].key, 0u, 0, 255);  // back to UNKNOWN
  slotOf_.erase(slots_[victim].key);
  slots_[victim].occupied = false;
  return victim;
}

void GpuBrickCache::upload(const BrickKey& key,
                           const std::uint8_t* borderedVoxels, bool pinned) {
  uploadFromBoundPbo(key, reinterpret_cast<std::size_t>(borderedVoxels),
                     pinned);
}

void GpuBrickCache::uploadFromBoundPbo(const BrickKey& key,
                                       std::size_t byteOffset, bool pinned,
                                       std::uint8_t minVal,
                                       std::uint8_t maxVal) {
  if (!pool_) return;
  if (auto it = slotOf_.find(key); it != slotOf_.end()) {
    slots_[it->second].lastTouched = frame_;
    return;
  }

  const std::uint32_t slot = acquireSlot(key);
  if (slot == UINT32_MAX) return;

  const std::uint32_t sx = slotDims_[2], sy = slotDims_[1];
  const std::uint32_t cx = slot % sx;
  const std::uint32_t cy = (slot / sx) % sy;
  const std::uint32_t cz = slot / (sx * sy);

  pool_->Bind();
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexSubImage3D(GL_TEXTURE_3D, 0, static_cast<GLint>(cx * borderedDim_),
                  static_cast<GLint>(cy * borderedDim_),
                  static_cast<GLint>(cz * borderedDim_), borderedDim_,
                  borderedDim_, borderedDim_, GL_RED, GL_UNSIGNED_BYTE,
                  reinterpret_cast<const void*>(byteOffset));

  slots_[slot] = Slot{key, true, pinned, frame_};
  slotOf_[key] = slot;
  writeEntry(key, kResidentBit | (cx << 0) | (cy << 10) | (cz << 20), minVal,
             maxVal);
}

void GpuBrickCache::syncPageTables() {
  for (auto& l : levels_) {
    if (!l.dirDirty || !l.dirTex) continue;
    l.dirTex->Bind();
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, l.dirDim[2], l.dirDim[1],
                    l.dirDim[0], GL_RED_INTEGER, GL_UNSIGNED_INT,
                    l.dirMirror.data());
    l.dirDirty = false;
  }

  if (!tilePoolTex_) return;
  bool boundTiles = false, boundOcc = false;
  const std::uint32_t px = tilePoolDims_[2], py = tilePoolDims_[1];
  for (std::uint32_t t = 0; t < tilesAllocated_; ++t) {
    if (!tileDirty_[t]) continue;
    const std::uint32_t tx = t % px;
    const std::uint32_t ty = (t / px) % py;
    const std::uint32_t tz = t / (px * py);
    if (!boundTiles) {
      tilePoolTex_->Bind();
      glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
      boundTiles = true;
    }
    glTexSubImage3D(GL_TEXTURE_3D, 0, static_cast<GLint>(tx * kTileDim),
                    static_cast<GLint>(ty * kTileDim),
                    static_cast<GLint>(tz * kTileDim), kTileDim, kTileDim,
                    kTileDim, GL_RED_INTEGER, GL_UNSIGNED_INT,
                    tileMirror_.data() + std::size_t{t} * kTileEntries);
  }
  for (std::uint32_t t = 0; t < tilesAllocated_; ++t) {
    if (!tileDirty_[t]) continue;
    const std::uint32_t tx = t % px;
    const std::uint32_t ty = (t / px) % py;
    const std::uint32_t tz = t / (px * py);
    if (!boundOcc) {
      occPoolTex_->Bind();
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
      boundOcc = true;
    }
    glTexSubImage3D(GL_TEXTURE_3D, 0, static_cast<GLint>(tx * kTileDim),
                    static_cast<GLint>(ty * kTileDim),
                    static_cast<GLint>(tz * kTileDim), kTileDim, kTileDim,
                    kTileDim, GL_RG, GL_UNSIGNED_BYTE,
                    occMirror_.data() + std::size_t{t} * kTileEntries * 2);
    tileDirty_[t] = false;
  }
}

}  // namespace sv::render
