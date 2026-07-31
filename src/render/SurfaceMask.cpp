#include "render/SurfaceMask.h"

#include <vtk_glew.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>

#include "core/Log.h"
#include "core/Profiling.h"

namespace sv::render {

void SurfaceMask::build(const data::TifXyzSurface& surface,
                        const std::array<std::uint64_t, 3>& shapeZyx,
                        float front, float behind, float sampleSpacing,
                        ThreadPool& pool,
                        const std::vector<std::array<std::uint32_t, 3>>&
                            brickGrids) {
  const auto t0 = std::chrono::steady_clock::now();
  grid_ = {std::uint32_t((shapeZyx[0] + 127) / 128),
           std::uint32_t((shapeZyx[1] + 127) / 128),
           std::uint32_t((shapeZyx[2] + 127) / 128)};

  const std::uint32_t W = surface.width(), H = surface.height();
  const auto& pts = surface.points();
  auto P = [&](std::uint32_t u, std::uint32_t v) {
    return pts[std::size_t{v} * W + u];
  };
  auto ok = [&](std::uint32_t u, std::uint32_t v) {
    return surface.valid(u, v);
  };

  // Estimate grid spacing in voxels (median-ish sample) to pick the
  // upsampling factor that lands swept samples <= sampleSpacing apart.
  double spacing = 0.0;
  int n = 0;
  for (std::uint32_t v = 0; v < H; v += 7)
    for (std::uint32_t u = 0; u + 1 < W; u += 7) {
      if (!ok(u, v) || !ok(u + 1, v)) continue;
      const auto a = P(u, v), b = P(u + 1, v);
      spacing += std::sqrt(double(b[0] - a[0]) * (b[0] - a[0]) +
                           double(b[1] - a[1]) * (b[1] - a[1]) +
                           double(b[2] - a[2]) * (b[2] - a[2]));
      ++n;
    }
  spacing = n ? spacing / n : 1.0;
  const int up =
      std::clamp(int(std::ceil(spacing / sampleSpacing)), 1, 32);
  const float wStep = sampleSpacing;

  const std::uint32_t gy = grid_[1], gx = grid_[2];
  const float maxZ = float(shapeZyx[0]) - 1.f;
  const float maxY = float(shapeZyx[1]) - 1.f;
  const float maxX = float(shapeZyx[2]) - 1.f;

  // Row-parallel sweep into per-task block maps, merged afterwards.
  const unsigned tasks = 8;
  std::vector<std::unordered_map<std::uint32_t, std::vector<std::uint8_t>>>
      partial(tasks);
  std::vector<std::future<void>> futs;
  for (unsigned ti = 0; ti < tasks; ++ti) {
    auto prom = std::make_shared<std::promise<void>>();
    futs.push_back(prom->get_future());
    pool.post([&, ti, prom](std::stop_token) {
      auto& local = partial[ti];
      auto mark = [&](float px, float py, float pz) {
        if (px < 0.f || py < 0.f || pz < 0.f || px > maxX || py > maxY ||
            pz > maxZ)
          return;
        const std::uint32_t x = std::uint32_t(px), y = std::uint32_t(py),
                            z = std::uint32_t(pz);
        const std::uint32_t brick =
            ((z >> 7) * gy + (y >> 7)) * gx + (x >> 7);
        auto& block = local[brick];
        if (block.empty()) block.assign(kBlockDim * kBlockDim * kBlockDim, 0);
        const std::uint32_t cz = (z >> 3) & 15, cy = (y >> 3) & 15,
                            cx = (x >> 3) & 15;
        block[(std::size_t{cz} * kBlockDim + cy) * kBlockDim + cx] = 255;
      };
      // Contiguous row bands per task: a brick is then touched by ~one
      // thread, so the per-thread maps stay disjoint (round-robin rows
      // duplicated every block in every map — memory blew up with thick
      // slabs).
      const std::uint32_t rows = H > 0 ? H - 1 : 0;
      const std::uint32_t v0 = rows * ti / tasks;
      const std::uint32_t v1 = rows * (ti + 1) / tasks;
      for (std::uint32_t v = v0; v < v1; ++v) {
        for (std::uint32_t u = 0; u + 1 < W; ++u) {
          if (!ok(u, v) || !ok(u + 1, v) || !ok(u, v + 1) || !ok(u + 1, v + 1))
            continue;
          const auto p00 = P(u, v), p10 = P(u + 1, v), p01 = P(u, v + 1),
                     p11 = P(u + 1, v + 1);
          for (int sv = 0; sv < up; ++sv)
            for (int su = 0; su < up; ++su) {
              const float fu = (float(su) + 0.5f) / float(up);
              const float fv = (float(sv) + 0.5f) / float(up);
              float base[3], du[3], dv[3];
              for (int a = 0; a < 3; ++a) {
                const float top = p00[a] + (p10[a] - p00[a]) * fu;
                const float bot = p01[a] + (p11[a] - p01[a]) * fu;
                base[a] = top + (bot - top) * fv;
                du[a] = (p10[a] - p00[a]) * (1.f - fv) +
                        (p11[a] - p01[a]) * fv;
                dv[a] = (p01[a] - p00[a]) * (1.f - fu) +
                        (p11[a] - p10[a]) * fu;
              }
              float nx = du[1] * dv[2] - du[2] * dv[1];
              float ny = du[2] * dv[0] - du[0] * dv[2];
              float nz = du[0] * dv[1] - du[1] * dv[0];
              const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
              if (len <= 0.f) continue;
              nx /= len, ny /= len, nz /= len;
              for (float w = -front; w <= behind; w += wStep)
                mark(base[0] + nx * w, base[1] + ny * w, base[2] + nz * w);
            }
        }
      }
      prom->set_value();
    });
  }
  for (auto& f : futs) f.get();

  blocks_.clear();
  for (auto& part : partial)
    for (auto& [brick, block] : part) {
      auto& dst = blocks_[brick];
      if (dst.empty()) {
        dst = std::move(block);
      } else {
        for (std::size_t i = 0; i < dst.size(); ++i)
          dst[i] = std::max(dst[i], block[i]);
      }
    }
  // Per-level brick presence: level 0 comes straight from the block keys
  // (a mask brick == a level-0 render brick); coarser levels OR their
  // children. Lets the raymarcher skip shell-free bricks at any LOD.
  presGrids_ = brickGrids;
  presence_.assign(brickGrids.size(), {});
  if (!brickGrids.empty()) {
    auto& p0 = presence_[0];
    p0.assign(std::size_t{grid_[0]} * grid_[1] * grid_[2], 0);
    for (const auto& [brick, block] : blocks_) p0[brick] = 255;
    for (std::size_t li = 1; li < brickGrids.size(); ++li) {
      const auto& cg = brickGrids[li - 1];   // child grid (z,y,x)
      const auto& g = brickGrids[li];
      auto& p = presence_[li];
      p.assign(std::size_t{g[0]} * g[1] * g[2], 0);
      const auto& c = presence_[li - 1];
      for (std::uint32_t z = 0; z < cg[0]; ++z)
        for (std::uint32_t y = 0; y < cg[1]; ++y)
          for (std::uint32_t x = 0; x < cg[2]; ++x) {
            if (!c[(std::size_t{z} * cg[1] + y) * cg[2] + x]) continue;
            const std::uint32_t pz = std::min(z / 2, g[0] - 1);
            const std::uint32_t py = std::min(y / 2, g[1] - 1);
            const std::uint32_t px = std::min(x / 2, g[2] - 1);
            p[(std::size_t{pz} * g[1] + py) * g[2] + px] = 255;
          }
    }
  }
  dirty_ = true;

  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
  const std::uint64_t mb =
      (std::uint64_t{blocks_.size()} * kBlockDim * kBlockDim * kBlockDim) >>
      20;
  logInfo("surface mask: {} blocks ({} MB) in {} ms (upsample {}x)",
          blocks_.size(), mb, ms, up);
  if (mb > 2048)
    logWarn("surface mask: very thick slab ({} MB of blocks); consider a "
            "smaller front/behind", mb);
}

void SurfaceMask::upload() {
  if (!dirty_) return;
  dirty_ = false;

  // Cube-ish pool shape for the block texture.
  const std::uint32_t count = std::uint32_t(blocks_.size());
  std::uint32_t px = std::max(1u, std::uint32_t(std::cbrt(double(count))));
  std::uint32_t py = px;
  while (px * py * ((count + px * py - 1) / (px * py)) < count) ++px;
  const std::uint32_t pz = (count + px * py - 1) / (px * py);
  poolDims_ = {std::max(1u, pz), py, px};

  std::vector<std::uint32_t> dir(
      std::size_t{grid_[0]} * grid_[1] * grid_[2], 0);
  const std::uint32_t bd = kBlockDim;
  std::vector<std::uint8_t> pool(std::size_t{poolDims_[0]} * poolDims_[1] *
                                     poolDims_[2] * bd * bd * bd,
                                 0);
  const std::uint32_t texW = poolDims_[2] * bd, texH = poolDims_[1] * bd;
  std::uint32_t slot = 0;
  for (const auto& [brick, block] : blocks_) {
    dir[brick] = slot + 1;
    const std::uint32_t sx = slot % poolDims_[2];
    const std::uint32_t sy = (slot / poolDims_[2]) % poolDims_[1];
    const std::uint32_t sz = slot / (poolDims_[2] * poolDims_[1]);
    for (std::uint32_t cz = 0; cz < bd; ++cz)
      for (std::uint32_t cy = 0; cy < bd; ++cy) {
        const std::size_t dstRow =
            (std::size_t{sz * bd + cz} * texH + (sy * bd + cy)) * texW +
            sx * bd;
        const std::size_t srcRow = (std::size_t{cz} * bd + cy) * bd;
        std::copy_n(block.data() + srcRow, bd, pool.data() + dstRow);
      }
    ++slot;
  }

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  if (!dirTex_) glGenTextures(1, &dirTex_);
  glBindTexture(GL_TEXTURE_3D, dirTex_);
  glTexImage3D(GL_TEXTURE_3D, 0, GL_R32UI, GLsizei(grid_[2]),
               GLsizei(grid_[1]), GLsizei(grid_[0]), 0, GL_RED_INTEGER,
               GL_UNSIGNED_INT, dir.data());
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  if (!poolTex_) glGenTextures(1, &poolTex_);
  glBindTexture(GL_TEXTURE_3D, poolTex_);
  glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, GLsizei(texW), GLsizei(texH),
               GLsizei(poolDims_[0] * bd), 0, GL_RED, GL_UNSIGNED_BYTE,
               pool.data());
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  presTex_.resize(presence_.size(), 0);
  for (std::size_t li = 0; li < presence_.size(); ++li) {
    if (!presTex_[li]) glGenTextures(1, &presTex_[li]);
    const auto& g = presGrids_[li];
    glBindTexture(GL_TEXTURE_3D, presTex_[li]);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R8, GLsizei(g[2]), GLsizei(g[1]),
                 GLsizei(g[0]), 0, GL_RED, GL_UNSIGNED_BYTE,
                 presence_[li].data());
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  }
  glBindTexture(GL_TEXTURE_3D, 0);
  logInfo("surface mask: uploaded {} blocks + {} presence levels to GPU",
          count, presTex_.size());
}

void SurfaceMask::releaseGraphicsResources() {
  if (dirTex_) {
    glDeleteTextures(1, &dirTex_);
    glDeleteTextures(1, &poolTex_);
    dirTex_ = poolTex_ = 0;
  }
  for (auto& t : presTex_)
    if (t) {
      glDeleteTextures(1, &t);
      t = 0;
    }
  dirty_ = built();
}

}  // namespace sv::render
