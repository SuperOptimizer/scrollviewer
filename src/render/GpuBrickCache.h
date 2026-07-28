#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include <vtkSmartPointer.h>
#include <vtkTextureObject.h>

#include "data/BrickKey.h"

class vtkOpenGLRenderWindow;

namespace sv::render {

// GPU residency for bordered bricks with SPARSE page tables:
//
//   brick pool   one large 3D R8 texture, slot-addressed (LRU eviction)
//   directory    per level, tiny R32UI texture; one entry per 16^3-brick
//                tile: 0 = UNKNOWN, 1 = ALL_EMPTY, else tileSlot + 2
//   tile pool    shared R32UI 3D texture of 16^3-entry page-table tiles,
//                allocated on demand (huge volumes stay ~MBs, not 100s of
//                MBs, because empty/unvisited space allocates nothing)
//   occ pool     RG8 twin of the tile pool: per-brick (min,max) density
//
// Page entry layout (32 bits): x:10 | y:10 | z:10 pool-slot coords,
// bit 30 = RESIDENT, bit 31 = EMPTY.
class GpuBrickCache {
 public:
  struct LevelDims {
    std::array<std::uint32_t, 3> grid;  // z,y,x brick-grid dims
  };

  static constexpr std::uint32_t kResidentBit = 1u << 30;
  static constexpr std::uint32_t kEmptyBit = 1u << 31;
  static constexpr std::uint32_t kTileDim = 16;
  // Directory entry values.
  static constexpr std::uint32_t kDirUnknown = 0;
  static constexpr std::uint32_t kDirAllEmpty = 1;

  GpuBrickCache(std::vector<LevelDims> levels, std::uint32_t borderedDim,
                std::uint64_t poolByteBudget);
  ~GpuBrickCache();

  void initialize(vtkOpenGLRenderWindow* renWin);
  bool initialized() const { return pool_ != nullptr; }
  void releaseGraphicsResources();

  void frameBegin(std::uint64_t frameIndex) { frame_ = frameIndex; }

  bool isResident(const data::BrickKey& key) const;
  void touch(const data::BrickKey& key);

  // Records a fill-value brick (page EMPTY flag, no pool slot).
  void markEmpty(const data::BrickKey& key);

  // Bulk manifest seeding: whole-empty tiles become a single ALL_EMPTY
  // directory entry (no tile allocation); mixed tiles allocate and get
  // per-brick EMPTY marks. isEmpty is called as (z, y, x).
  void seedEmpties(
      int level,
      const std::function<bool(std::uint32_t, std::uint32_t, std::uint32_t)>&
          isEmpty);

  void upload(const data::BrickKey& key, const std::uint8_t* borderedVoxels,
              bool pinned = false);
  void uploadFromBoundPbo(const data::BrickKey& key, std::size_t byteOffset,
                          bool pinned = false, std::uint8_t minVal = 0,
                          std::uint8_t maxVal = 255);

  void syncPageTables();

  vtkTextureObject* poolTexture() { return pool_; }
  vtkTextureObject* directoryTexture(int level) { return levels_[level].dirTex; }
  vtkTextureObject* tilePoolTexture() { return tilePoolTex_; }
  vtkTextureObject* occPoolTexture() { return occPoolTex_; }

  int levelCount() const { return static_cast<int>(levels_.size()); }
  LevelDims levelDims(int level) const { return {levels_[level].grid}; }
  std::array<std::uint32_t, 3> poolSlotDims() const { return slotDims_; }
  std::array<std::uint32_t, 3> tilePoolDims() const { return tilePoolDims_; }
  std::uint32_t borderedDim() const { return borderedDim_; }
  std::size_t slotCount() const { return slots_.size(); }
  std::size_t residentCount() const { return slotOf_.size(); }
  std::size_t tilesAllocated() const { return tilesAllocated_; }

 private:
  struct Slot {
    data::BrickKey key;
    bool occupied = false;
    bool pinned = false;
    std::uint64_t lastTouched = 0;
  };

  struct Level {
    std::array<std::uint32_t, 3> grid;    // brick grid (z,y,x)
    std::array<std::uint32_t, 3> dirDim;  // ceil(grid / kTileDim) (z,y,x)
    std::vector<std::uint32_t> dirMirror;
    vtkSmartPointer<vtkTextureObject> dirTex;
    bool dirDirty = false;
  };

  std::size_t dirIndex(const Level& l, std::uint32_t z, std::uint32_t y,
                       std::uint32_t x) const {
    return (std::size_t{z / kTileDim} * l.dirDim[1] + y / kTileDim) *
               l.dirDim[2] +
           x / kTileDim;
  }
  std::size_t inTileIndex(std::uint32_t z, std::uint32_t y,
                          std::uint32_t x) const {
    return (std::size_t{z % kTileDim} * kTileDim + y % kTileDim) * kTileDim +
           x % kTileDim;
  }

  // Returns current entry value; 0 (UNKNOWN) when no tile is allocated.
  std::uint32_t readEntry(const data::BrickKey& key) const;
  // Ensures a tile exists for the key's region (converting ALL_EMPTY into a
  // materialized tile of EMPTY entries) and writes the entry + occupancy.
  void writeEntry(const data::BrickKey& key, std::uint32_t value,
                  std::uint8_t occMin, std::uint8_t occMax);
  std::uint32_t allocateTile(Level& l, std::size_t dirIdx);
  std::uint32_t acquireSlot(const data::BrickKey& key);

  std::vector<Level> levels_;
  std::uint32_t borderedDim_;
  std::uint64_t poolBudget_;
  std::array<std::uint32_t, 3> slotDims_{};

  vtkOpenGLRenderWindow* renWin_ = nullptr;
  vtkSmartPointer<vtkTextureObject> poolHolder_;
  vtkTextureObject* pool_ = nullptr;
  vtkSmartPointer<vtkTextureObject> tilePoolTex_;
  vtkSmartPointer<vtkTextureObject> occPoolTex_;

  // Tile storage mirrors: tile t occupies [t*4096, (t+1)*4096) entries.
  std::array<std::uint32_t, 3> tilePoolDims_{};  // tiles per axis (z,y,x)
  std::uint32_t tileCapacity_ = 0;
  std::uint32_t tilesAllocated_ = 0;
  std::vector<std::uint32_t> tileMirror_;
  std::vector<std::uint8_t> occMirror_;  // 2 bytes per entry
  std::vector<bool> tileDirty_;

  std::vector<Slot> slots_;
  std::vector<std::uint32_t> freeSlots_;
  std::unordered_map<data::BrickKey, std::uint32_t, data::BrickKeyHash> slotOf_;
  std::uint64_t frame_ = 0;
};

}  // namespace sv::render
