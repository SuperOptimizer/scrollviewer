#include "render/vtkScrollVolumeMapper.h"

#include <vtk_glew.h>

#include <vtkCamera.h>
#include <vtkImageData.h>
#include <vtkInformation.h>
#include <vtkMath.h>
#include <vtkMatrix4x4.h>
#include <vtkObjectFactory.h>
#include <vtkOpenGLQuadHelper.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkOpenGLShaderCache.h>
#include <vtkOpenGLState.h>
#include <vtkRenderer.h>
#include <vtkShaderProgram.h>
#include <vtkTextureObject.h>
#include <vtkVolume.h>

#include <algorithm>
#include <cmath>
#include <format>
#include <unordered_set>

#include "core/Log.h"
#include "core/Profiling.h"

namespace sv::render {

vtkStandardNewMacro(vtkScrollVolumeMapper);

vtkScrollVolumeMapper::vtkScrollVolumeMapper() {
  // vtkVolume::RenderVolumetricGeometry silently skips mappers whose
  // GetDataObjectInput() is null, so our Render() would never run. The real
  // dataset is the virtual brick cache; feed the pipeline a token 1-voxel
  // image so prop dispatch reaches us.
  vtkNew<vtkImageData> dummy;
  dummy->SetDimensions(1, 1, 1);
  dummy->AllocateScalars(VTK_UNSIGNED_CHAR, 1);
  this->SetInputData(dummy);
}

int vtkScrollVolumeMapper::FillInputPortInformation(int port,
                                                    vtkInformation* info) {
  // The dataset is the virtual brick cache, not a vtkImageData connection;
  // keep the inherited port (the volume pipeline queries it unconditionally)
  // but mark it optional so an empty connection is not an error.
  if (!this->Superclass::FillInputPortInformation(port, info)) return 0;
  info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 1);
  return 1;
}
vtkScrollVolumeMapper::~vtkScrollVolumeMapper() = default;

void vtkScrollVolumeMapper::SetVolumeExtent(
    const std::array<std::uint64_t, 3>& shapeZyx, double spacing) {
  ShapeZyx = shapeZyx;
  Spacing = spacing;
  // VTK bounds are (xmin,xmax, ymin,ymax, zmin,zmax) in world units.
  Bounds[0] = 0;
  Bounds[1] = double(ShapeZyx[2]) * Spacing;
  Bounds[2] = 0;
  Bounds[3] = double(ShapeZyx[1]) * Spacing;
  Bounds[4] = 0;
  Bounds[5] = double(ShapeZyx[0]) * Spacing;
}

double* vtkScrollVolumeMapper::GetBounds() { return Bounds; }

void vtkScrollVolumeMapper::SetOverlay(
    GpuBrickCache* cache, const std::array<std::uint64_t, 3>& shapeZyx) {
  if ((Overlay != nullptr) != (cache != nullptr)) Quad.reset();
  Overlay = cache;
  OverlayShapeZyx = shapeZyx;
}

void vtkScrollVolumeMapper::SetDensityMask(const SurfaceMask* mask) {
  if ((Mask != nullptr) != (mask != nullptr)) Quad.reset();  // shader variant
  Mask = mask;
}

void vtkScrollVolumeMapper::EnsureLowResTarget(int w, int h) {
  if (LowResFbo && LowResW == w && LowResH == h) return;
  if (!LowResFbo) {
    glGenFramebuffers(1, &LowResFbo);
    glGenTextures(1, &LowResTex);
  }
  glBindTexture(GL_TEXTURE_2D, LowResTex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_2D, 0);
  glBindFramebuffer(GL_FRAMEBUFFER, LowResFbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         LowResTex, 0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  LowResW = w;
  LowResH = h;
}

void vtkScrollVolumeMapper::ReleaseGraphicsResources(vtkWindow*) {
  Quad.reset();
  UpscaleQuad.reset();
  if (SurfVao) {
    glDeleteVertexArrays(1, &SurfVao);
    glDeleteBuffers(1, &SurfVbo);
    glDeleteBuffers(1, &SurfIbo);
    SurfVao = SurfVbo = SurfIbo = 0;
    SurfIndexCount = 0;
    SurfaceDirty = Surface != nullptr;
  }
  if (ClusterVao) {
    glDeleteVertexArrays(1, &ClusterVao);
    glDeleteBuffers(1, &ClusterVbo);
    ClusterVao = ClusterVbo = 0;
    ClustersDirty = !Clusters.empty();
  }
  if (LowResFbo) {
    glDeleteFramebuffers(1, &LowResFbo);
    glDeleteTextures(1, &LowResTex);
    LowResFbo = LowResTex = 0;
    LowResW = LowResH = 0;
  }
  if (FeedbackSsbo[0]) {
    for (int i = 0; i < 2; ++i) {
      if (FeedbackFence[i]) {
        glDeleteSync(static_cast<GLsync>(FeedbackFence[i]));
        FeedbackFence[i] = nullptr;
      }
      if (FeedbackMapped[i]) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, FeedbackSsbo[i]);
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        FeedbackMapped[i] = nullptr;
      }
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glDeleteBuffers(2, FeedbackSsbo);
    FeedbackSsbo[0] = FeedbackSsbo[1] = 0;
  }
}

void vtkScrollVolumeMapper::InitFeedbackBuffers() {
  glGenBuffers(2, FeedbackSsbo);
  const std::size_t bytes = 16 + std::size_t{kMaxFeedback} * 8;
  const GLbitfield flags = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT |
                           GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
  for (int i = 0; i < 2; ++i) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, FeedbackSsbo[i]);
    glBufferStorage(GL_SHADER_STORAGE_BUFFER, bytes, nullptr, flags);
    FeedbackMapped[i] = static_cast<std::uint32_t*>(
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, bytes, flags));
    FeedbackMapped[i][0] = 0;
  }
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void vtkScrollVolumeMapper::ReadFeedback(int index) {
  // Zero-stall: only read once the frame that wrote this buffer has
  // retired; otherwise leave it for the next harvest.
  if (FeedbackFence[index]) {
    const GLenum r = glClientWaitSync(
        static_cast<GLsync>(FeedbackFence[index]), 0, 0);
    if (r != GL_ALREADY_SIGNALED && r != GL_CONDITION_SATISFIED) return;
    glDeleteSync(static_cast<GLsync>(FeedbackFence[index]));
    FeedbackFence[index] = nullptr;
  }

  std::uint32_t* buf = FeedbackMapped[index];
  if (!buf) return;
  const std::uint32_t count = std::min(buf[0], kMaxFeedback);
  if (count > 0) {
    const std::uint32_t* raw = buf + 4;  // past counter + padding
    std::unordered_set<std::uint64_t> seen;
    FeedbackRequests.clear();
    for (std::uint32_t i = 0; i < count; ++i) {
      const std::uint32_t level = raw[i * 2];
      const std::uint32_t zyx = raw[i * 2 + 1];
      const std::uint64_t packed = (std::uint64_t{level} << 32) | zyx;
      if (!seen.insert(packed).second) continue;
      FeedbackRequests.push_back(data::ChunkCoord{
          static_cast<std::uint8_t>(level), (zyx >> 20) & 0x3FFu,
          (zyx >> 10) & 0x3FFu, zyx & 0x3FFu});
    }
  }
  buf[0] = 0;  // reset counter for the buffer's next write cycle
}

// GLSL shared by the raymarch and surface programs: streaming request
// buffer, brick pool + sparse page-table plumbing, and every uniform both
// passes read.
std::string vtkScrollVolumeMapper::BuildCommonGlsl() const {
  const int levels = Cache ? Cache->levelCount() : 1;
  // Per-level page-table samplers are distinct uniforms (GLSL cannot index a
  // sampler array with a non-uniform value); a switch helper dispatches.
  std::string dirDecls, dirCases;
  for (int i = 0; i < levels; ++i) {
    dirDecls += std::format("uniform usampler3D pageDir{};\n", i);
    dirCases += std::format(
        "    case {}: return texelFetch(pageDir{}, c, 0).r;\n", i, i);
  }

  return std::format(R"glsl(
// Ray-guided streaming: bricks wanted at a finer level than was resident.
layout(std430, binding = 0) buffer RequestBuf {{
  uint reqCount;
  uint reqPad0, reqPad1, reqPad2;
  uvec2 reqs[];  // (level, z<<20|y<<10|x)
}};

const int kLevels = {1};
const uint kMaxReq = {3}u;

uniform sampler3D brickPool;
{0}
uniform vec3 volSizeWorld;     // full-res extent * spacing (x,y,z)
uniform vec3 levelShape[{1}];  // voxels per level (x,y,z)
uniform vec3 gridDim[{1}];     // brick-grid dims per level (x,y,z)
uniform vec3 poolTexels;       // pool texture size in texels
uniform vec3 invPoolTexels;    // 1 / poolTexels (divide-free tap coords)
uniform float chunkDim;        // payload voxels per brick axis
uniform float borderedDim;     // chunkDim + 2
uniform float desiredLevel;    // minimum (finest) level to sample
uniform float sampleStepScale;
uniform float wppConst;        // world-per-pixel, constant part (ortho)
uniform float wppPerDist;      // world-per-pixel per unit distance (persp)
uniform float voxelWorld0;     // world size of a level-0 voxel
uniform float winWidth;        // window/level on normalized value
uniform float winCenter;
uniform float opacityScale;
uniform float occSkipThreshold;  // brick max density <= this => invisible
uniform usampler3D tilePool;   // shared 16^3-entry page-table tiles
uniform sampler3D  occPool;    // RG8 twin of tilePool: per-brick min/max
uniform int tilePoolDimX;
uniform int tilePoolDimY;

// Sparse page tables: per-level directory (one entry per 16^3-brick tile)
// -> shared tile pool. Directory: 0 = unknown, 1 = whole tile empty,
// else tileSlot + 2.
uint fetchDir(int level, ivec3 c) {{
  switch (level) {{
{2}  }}
  return 0u;
}}

ivec3 tileTexel(uint t, ivec3 brick) {{
  int ti = int(t);
  ivec3 tileCoord = ivec3(ti % tilePoolDimX,
                          (ti / tilePoolDimX) % tilePoolDimY,
                          ti / (tilePoolDimX * tilePoolDimY));
  return tileCoord * 16 + (brick & ivec3(15));
}}

uint fetchPageEntry(int level, ivec3 brick) {{
  uint d = fetchDir(level, brick >> 4);
  if (d < 2u) return (d == 1u) ? 0x80000000u : 0u;
  return texelFetch(tilePool, tileTexel(d - 2u, brick), 0).r;
}}
)glsl",
                     dirDecls, levels, dirCases, kMaxFeedback);
}

std::string vtkScrollVolumeMapper::BuildFragmentShader() const {
  // Optional shell-mask variant: density outside the surface mask is zero.
  // Cell/brick shifts assume 128^3 chunks and 8-voxel mask cells
  // (SurfaceMask::kCellVox).
  const bool hasMask = Mask != nullptr;
  std::string maskLvlDecls, maskLvlCases;
  if (hasMask) {
    const int mlv = Cache ? Cache->levelCount() : 1;
    for (int i = 0; i < mlv; ++i) {
      maskLvlDecls += std::format("uniform sampler3D maskLvl{};\n", i);
      maskLvlCases += std::format(
          "    case {}: return texelFetch(maskLvl{}, b, 0).r;\n", i, i);
    }
  }
  const std::string maskDecls =
      hasMask ? std::format(R"glsl(
uniform usampler3D maskDir;   // level-0 brick grid: 0 = outside shell
uniform sampler3D maskPool;   // 16^3-cell blocks, 8-voxel cells
uniform int maskPoolDimX;
uniform int maskPoolDimY;
uniform float maskStrength;   // surface-shell highlight blend
uniform float maskIsolate;    // 1 = show only the shell (no tint)
{0}
float maskAt(ivec3 v0) {{
  uint d = texelFetch(maskDir, v0 >> 7, 0).r;
  if (d == 0u) return 0.0;
  int s = int(d - 1u);
  ivec3 sc = ivec3(s % maskPoolDimX, (s / maskPoolDimX) % maskPoolDimY,
                   s / (maskPoolDimX * maskPoolDimY));
  return texelFetch(maskPool, sc * 16 + ((v0 >> 3) & 15), 0).r;
}}

// Per-LOD brick presence: does the shell touch this brick at all?
float maskPresent(int level, ivec3 b) {{
  switch (level) {{
{1}  }}
  return 1.0;
}}
)glsl",
                            maskLvlDecls, maskLvlCases)
              : "";
  // Two mask modes, switched by uniform (no shader rebuild): highlight
  // tints the shell cyan; isolate zeroes density outside it, leaving only
  // the raw CT (and any overlay) that intersects the slab.
  const std::string maskWeight =
      hasMask ? "        float mk = maskAt(ivec3(clamp(((ro + t * rd) / "
                "volSizeWorld) * levelShape[0], vec3(0.0), levelShape[0] - "
                "1.0)));\n"
                "        w *= mix(1.0, mk, maskIsolate);\n"
              : "";
  const std::string maskTint =
      hasMask ? "        col = mix(col, vec3(0.25, 0.85, 1.0),\n"
                "                  maskStrength * mk * (1.0 - maskIsolate));\n"
              : "";

  // Prediction-overlay variant: an independent second volume (own page
  // tables and pool) sampled at each visible sample; values above the
  // threshold tint the sample toward a hot color.
  std::string ovDecls, ovSample;
  if (Overlay) {
    const int ovLevels = Overlay->levelCount();
    std::string ovDirDecls, ovDirCases;
    for (int i = 0; i < ovLevels; ++i) {
      ovDirDecls += std::format("uniform usampler3D ovPageDir{};\n", i);
      ovDirCases += std::format(
          "    case {}: return texelFetch(ovPageDir{}, c, 0).r;\n", i, i);
    }
    ovDecls = std::format(R"glsl(
uniform sampler3D ovBrickPool;
{0}
uniform usampler3D ovTilePool;
uniform int ovTilePoolDimX;
uniform int ovTilePoolDimY;
uniform vec3 ovLevelShape[{1}];
uniform vec3 ovGridDim[{1}];
uniform vec3 ovPoolTexels;
uniform float ovChunkDim;
uniform float ovBorderedDim;
uniform int ovLevelOffset;   // main lod -> overlay lod shift
uniform float ovStrength;
uniform float ovThreshold;
const int kOvLevels = {1};

uint ovFetchDir(int level, ivec3 c) {{
  switch (level) {{
{2}  }}
  return 0u;
}}

uint ovFetchPageEntry(int level, ivec3 brick) {{
  uint d = ovFetchDir(level, brick >> 4);
  if (d < 2u) return (d == 1u) ? 0x80000000u : 0u;
  int ti = int(d - 2u);
  ivec3 tc = ivec3(ti % ovTilePoolDimX, (ti / ovTilePoolDimX) % ovTilePoolDimY,
                   ti / (ovTilePoolDimX * ovTilePoolDimY));
  return texelFetch(ovTilePool, tc * 16 + (brick & ivec3(15)), 0).r;
}}

float ovSampleAt(vec3 pn, int lod) {{
  for (int i = lod; i < kOvLevels; ++i) {{
    vec3 v = pn * ovLevelShape[i];
    ivec3 b = ivec3(clamp(floor(v / ovChunkDim), vec3(0.0),
                          ovGridDim[i] - 1.0));
    uint pe = ovFetchPageEntry(i, b);
    if ((pe & 0x80000000u) != 0u) return 0.0;
    if ((pe & 0x40000000u) != 0u) {{
      vec3 poolBase = vec3(float(pe & 0x3FFu), float((pe >> 10) & 0x3FFu),
                           float((pe >> 20) & 0x3FFu)) * ovBorderedDim + 1.0;
      vec3 inBrick = clamp(v - vec3(b) * ovChunkDim, vec3(0.0),
                           vec3(ovChunkDim));
      return texture(ovBrickPool, (poolBase + inBrick) / ovPoolTexels).r;
    }}
  }}
  return 0.0;
}}
)glsl",
                          ovDirDecls, ovLevels, ovDirCases);
    // Gated on visible density: transparent samples can't show a tint, so
    // skip the overlay page walk entirely there (most of empty space).
    ovSample =
        "        if (w > 0.004) {\n"
        "        float ip = ovSampleAt((ro + t * rd) / volSizeWorld,\n"
        "            clamp(L - ovLevelOffset, 0, kOvLevels - 1));\n"
        "        if (ip > ovThreshold)\n"
        "          col = mix(col, vec3(1.0, 0.30, 0.05),\n"
        "                    min(1.0, (ip - ovThreshold) / max(0.001, 1.0 - "
        "ovThreshold)) * ovStrength);\n"
        "        }\n";
  }

  return std::format(R"glsl(//VTK::System::Dec
#extension GL_ARB_shader_storage_buffer_object : require
in vec2 texCoord;
out vec4 fragColor;
{0}
{1}
{4}
uniform mat4 invViewProj;      // NDC -> world
uniform int renderMode;        // 0 x-ray, 1 shaded DVR, 2 isosurface
uniform int filterOp;          // 0 none, 1 smooth, 2 sharpen, 3 edges
uniform float filterAmt;       // filter strength
uniform float filterFloor;     // high-pass: density below this reads as 0
uniform vec3 lightDir;         // world direction toward the key light
uniform float lightAmbient;    // ambient/fill scale
uniform int shadowsOn;         // surface-mode shadow rays

// Central-difference density gradient in pool texel space. gp is
// poolBase + inBrick; +-1 texel stays inside this brick's bordered block,
// so the taps are valid right up to the brick faces.
vec3 poolGrad(vec3 gp) {{
  vec3 e = vec3(1.0, 0.0, 0.0);
  return vec3(texture(brickPool, (gp + e.xyy) * invPoolTexels).r -
                  texture(brickPool, (gp - e.xyy) * invPoolTexels).r,
              texture(brickPool, (gp + e.yxy) * invPoolTexels).r -
                  texture(brickPool, (gp - e.yxy) * invPoolTexels).r,
              texture(brickPool, (gp + e.yyx) * invPoolTexels).r -
                  texture(brickPool, (gp - e.yyx) * invPoolTexels).r);
}}

// Per-sample 3D processing, evaluated in pool texel space so it works on
// whatever LOD the ray is marching (taps stay within the bordered brick).
// One 6-neighbor stencil serves filter AND lighting: the same +-1 taps
// yield the box blur (sum) and the central-difference gradient
// (differences), so shaded+filtered modes pay 6 taps, not 12. `grad` is
// the raw-density gradient (valid whenever filterOp != 0).
float filteredDensity(float dens, vec3 gp, out vec3 grad) {{
  grad = vec3(0.0);
  if (filterOp != 0) {{
    vec3 e = vec3(1.0, 0.0, 0.0);
    float xp = texture(brickPool, (gp + e.xyy) * invPoolTexels).r;
    float xm = texture(brickPool, (gp - e.xyy) * invPoolTexels).r;
    float yp = texture(brickPool, (gp + e.yxy) * invPoolTexels).r;
    float ym = texture(brickPool, (gp - e.yxy) * invPoolTexels).r;
    float zp = texture(brickPool, (gp + e.yyx) * invPoolTexels).r;
    float zm = texture(brickPool, (gp - e.yyx) * invPoolTexels).r;
    grad = vec3(xp - xm, yp - ym, zp - zm);
    if (filterOp == 3) {{
      // Edge field: gradient magnitude replaces density (fibre/boundary
      // visualization; also isosurfaces nicely in surface mode).
      dens = clamp(length(grad) * (1.0 + 6.0 * filterAmt), 0.0, 1.0);
    }} else {{
      float blur = (xp + xm + yp + ym + zp + zm + 2.0 * dens) * 0.125;
      dens = (filterOp == 1)
                 ? mix(dens, blur, clamp(filterAmt, 0.0, 1.0))     // smooth
                 : clamp(dens + filterAmt * (dens - blur), 0.0, 1.0);  // unsharp
    }}
  }}
  // High-pass floor: below-threshold voxels are hard zero.
  return (dens < filterFloor) ? 0.0 : dens;
}}

// Warm papyrus ramp for the shaded modes: umber -> tan -> ivory.
vec3 densRamp(float w) {{
  vec3 c0 = vec3(0.13, 0.08, 0.05);
  vec3 c1 = vec3(0.62, 0.44, 0.26);
  vec3 c2 = vec3(0.96, 0.87, 0.70);
  return w < 0.5 ? mix(c0, c1, w * 2.0) : mix(c1, c2, w * 2.0 - 1.0);
}}

// Resolve one density sample anywhere in the volume through the page-table
// hierarchy (coarser-level fallback). Used off the primary marching path,
// e.g. shadow rays.
float densAtLod(vec3 pn, int lod) {{
  for (int i = lod; i < kLevels; ++i) {{
    vec3 v = pn * levelShape[i];
    ivec3 b = ivec3(clamp(floor(v / chunkDim), vec3(0.0), gridDim[i] - 1.0));
    uint pe = fetchPageEntry(i, b);
    if ((pe & 0x80000000u) != 0u) return 0.0;
    if ((pe & 0x40000000u) != 0u) {{
      vec3 pb = vec3(float(pe & 0x3FFu), float((pe >> 10) & 0x3FFu),
                     float((pe >> 20) & 0x3FFu)) * borderedDim + 1.0;
      vec3 ib = clamp(v - vec3(b) * chunkDim, vec3(0.0), vec3(chunkDim));
      return texture(brickPool, (pb + ib) * invPoolTexels).r;
    }}
  }}
  return 0.0;
}}

// Soft shadow: transmittance marched toward the light at one-coarser LOD,
// with geometric step growth (tight contact shadows, cheap distant ones).
// One march per pixel (surface mode only).
float shadowFactor(vec3 pw, int L) {{
  if (shadowsOn == 0) return 1.0;
  int Ls = min(L + 1, kLevels - 1);
  float s = voxelWorld0 * exp2(float(Ls)) * 3.0;  // skip self-occlusion
  float trans = 1.0;
  // 16 steps at 1.35x growth span the same distance as 24 at 1.22x; the
  // per-step density weight is scaled up to keep total extinction close.
  for (int i = 0; i < 16 && trans > 0.05; ++i) {{
    vec3 pn = (pw + lightDir * s) / volSizeWorld;
    if (any(lessThan(pn, vec3(0.0))) || any(greaterThan(pn, vec3(1.0)))) break;
    float d = densAtLod(pn, Ls);
    float wq = clamp((d - winCenter) / winWidth + 0.5, 0.0, 1.0);
    trans *= 1.0 - wq * 0.42;
    s *= 1.35;
  }}
  return max(trans, 0.12);  // never pitch black
}}

// Cook-Torrance GGX (roughness 0.5, dielectric F0) under the directional
// key light, plus fill and hemispheric ambient scaled by lightAmbient. The
// volume is axis-aligned, so voxel gradients are already world-space normals.
vec3 shadeSurface(vec3 albedo, vec3 N, vec3 rd, float shadow) {{
  vec3 V = -rd;
  vec3 Lk = lightDir;
  vec3 H = normalize(Lk + V);
  float ndl = max(dot(N, Lk), 0.0);
  float ndv = max(dot(N, V), 0.0);
  float ndh = max(dot(N, H), 0.0);
  const float a2 = 0.0625;  // roughness^4
  float den = ndh * ndh * (a2 - 1.0) + 1.0;
  float D = a2 / (3.14159 * den * den);
  float F = 0.04 + 0.96 * pow(1.0 - max(dot(H, V), 0.0), 5.0);
  const float k = 0.28;  // (rough+1)^2 / 8
  float G = (ndv / (ndv * (1.0 - k) + k)) * (ndl / (ndl * (1.0 - k) + k));
  float spec = D * F * G / max(4.0 * ndv * ndl, 0.001) * ndl;
  vec3 amb = albedo * (0.16 + 0.08 * N.y) * lightAmbient;
  vec3 fill = albedo * 0.18 * lightAmbient * max(dot(N, -Lk), 0.0);
  return amb + fill + (albedo * ndl * 0.85 + vec3(spec)) * shadow;
}}

bool rayBox(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax, out float t0, out float t1) {{
  vec3 inv = 1.0 / rd;
  vec3 tlo = (bmin - ro) * inv;
  vec3 thi = (bmax - ro) * inv;
  vec3 tmin = min(tlo, thi), tmax = max(tlo, thi);
  t0 = max(max(tmin.x, tmin.y), tmin.z);
  t1 = min(min(tmax.x, tmax.y), tmax.z);
  return t1 > max(t0, 0.0);
}}

// Cheap hash-based jitter to hide stepping bands.
float jitter(vec2 co) {{
  return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}}

void main() {{
  vec2 ndc = texCoord * 2.0 - 1.0;
  vec4 nearW = invViewProj * vec4(ndc, -1.0, 1.0);
  vec4 farW  = invViewProj * vec4(ndc,  1.0, 1.0);
  vec3 ro = nearW.xyz / nearW.w;
  vec3 rd = normalize(farW.xyz / farW.w - ro);

  float t0, t1;
  if (!rayBox(ro, rd, vec3(0.0), volSizeWorld, t0, t1)) discard;
  t0 = max(t0, 0.0);

  int minLod = int(clamp(desiredLevel, 0.0, float(kLevels - 1)));
  vec3 invRd = 1.0 / rd;

  float dtMin = voxelWorld0 * exp2(float(minLod)) * sampleStepScale;
  float jit = jitter(gl_FragCoord.xy);
  float t = t0 + jit * dtMin;
  vec4 acc = vec4(0.0);
  // At most one streaming request per fragment, and only ~1/8 of fragments
  // request per frame: 2M same-address atomicAdds serialize badly, and a
  // decimated request stream covers the same bricks over a few frames
  // (the pipeline dedupes). kMaxReq caps the useful count anyway.
  bool requested = jit > 0.125;

  // Brick-marching: resolve the page table once per brick, then inner-loop
  // through the brick's samples straight from the pool texture.
  for (int outer = 0; outer < 1024 && t < t1 && acc.a < 0.96; ++outer) {{
    // Cone LOD at the brick entry point (GigaVoxels-style), never finer
    // than the frame's minimum.
    float wpp = wppConst + wppPerDist * t;
    // Opacity-adaptive LOD: behind ~2/3 accumulated opacity the remaining
    // contribution is small enough that one-level-coarser sampling is
    // invisible — and it also stops streaming requests for occluded bricks.
    int lod = clamp(int(floor(log2(max(1.0, wpp / voxelWorld0)))) +
                        int(acc.a * 2.0),
                    minLod, kLevels - 1);
    vec3 p = (ro + t * rd) / volSizeWorld;

    // Fallback walk. EMPTY entries chain upward: the coarsest contiguous
    // empty level wins so one jump can cross a huge region (a deep-zoom ray
    // otherwise crawls across hundreds of fine empty bricks). RESIDENT stops
    // the walk (finer empties still take precedence over coarser content).
    int L = -1;
    uint e = 0u;
    ivec3 brick = ivec3(0);
    bool tileJump = false;  // empty came from a whole-tile ALL_EMPTY entry
    uint dSel = 0u;         // directory entry of the selected level
    for (int i = lod; i < kLevels; ++i) {{
      vec3 v = p * levelShape[i];
      ivec3 b = ivec3(clamp(floor(v / chunkDim), vec3(0.0), gridDim[i] - 1.0));
      uint d = fetchDir(i, b >> 4);
      uint pe = (d == 0u) ? 0u
              : (d == 1u) ? 0x80000000u
              : texelFetch(tilePool, tileTexel(d - 2u, b), 0).r;
      if ((pe & 0x80000000u) != 0u) {{
        L = i;
        e = pe;
        brick = b;
        tileJump = (d == 1u);
        dSel = d;
        continue;  // empty: try to widen the jump at a coarser level
      }}
      if ((pe & 0x40000000u) != 0u) {{
        if (L < 0) {{
          L = i;
          e = pe;
          brick = b;
          dSel = d;
        }}
        break;
      }}
      if (L >= 0) break;  // unknown above an empty: jump what we have
    }}
    if (L < 0) {{  // nothing known yet (startup only)
      t += voxelWorld0 * exp2(float(lod)) * sampleStepScale;
      continue;
    }}

    float dt = voxelWorld0 * exp2(float(L)) * sampleStepScale;
    // True brick extent: chunkDim voxels at this level's voxel size. Using
    // volSize/gridDim would be the AVERAGE brick size, which drifts away
    // from page-table indexing because edge bricks are padded.
    vec3 brickWorld = chunkDim * (volSizeWorld / levelShape[L]);
    vec3 bmin = vec3(brick) * brickWorld;
    vec3 farSide = mix(bmin + brickWorld, bmin, lessThan(rd, vec3(0.0)));
    float tExit = min(min((farSide.x - ro.x) * invRd.x,
                          (farSide.y - ro.y) * invRd.y),
                      (farSide.z - ro.z) * invRd.z);

    // Skip bricks that are empty or entirely below the visibility window.
    bool emptyBrick = (e & 0x80000000u) != 0u;
    if (!emptyBrick && dSel >= 2u) {{
      // dSel was fetched during the walk: index the occ pool directly.
      vec2 mm = texelFetch(occPool, tileTexel(dSel - 2u, brick), 0).rg;
      emptyBrick = mm.y <= occSkipThreshold;
      // Mode-aware culls from the same min/max pair:
      //  - edge field: spread bounds the gradient (sqrt(3) for the vector
      //    norm); a near-uniform brick has no edges to show.
      //  - isosurface: a brick whose max (sharpened upper bound if the
      //    unsharp filter is on) stays below the iso value has no crossing.
      float sp = (mm.y - mm.x) * (1.0 + 6.0 * filterAmt) * 1.74;
      if (filterOp == 3)
        emptyBrick = emptyBrick || sp <= max(filterFloor, 0.002);
      if (renderMode == 2) {{
        float dmax = (filterOp == 3) ? sp
                     : (filterOp == 2) ? mm.y * (1.0 + filterAmt)
                                       : mm.y;
        emptyBrick = emptyBrick || dmax < winCenter;
      }}
    }}
    if (emptyBrick) {{
      if (tileJump) {{
        // The whole 16^3-brick tile is known empty: exit the tile AABB in
        // one jump instead of crawling across it brick by brick.
        vec3 tileWorld = 16.0 * brickWorld;
        vec3 tmin = vec3(brick >> 4) * tileWorld;
        vec3 tfar = mix(tmin + tileWorld, tmin, lessThan(rd, vec3(0.0)));
        tExit = min(min((tfar.x - ro.x) * invRd.x,
                        (tfar.y - ro.y) * invRd.y),
                    (tfar.z - ro.z) * invRd.z);
      }}
      t = max(tExit, t) + dt * 0.05;
      continue;
    }}
{2}
    if (L > lod && !requested) {{
      // Sampling coarser than wanted: ask for the finer brick. Plain read
      // first — once the buffer is full the atomic would be pure contention.
      requested = true;
      if (reqCount < kMaxReq) {{
        vec3 vw = p * levelShape[lod];
        ivec3 want = ivec3(clamp(floor(vw / chunkDim), vec3(0.0),
                                 gridDim[lod] - 1.0));
        uint idx = atomicAdd(reqCount, 1u);
        if (idx < kMaxReq)
          reqs[idx] = uvec2(uint(lod), (uint(want.z) << 20) |
                                       (uint(want.y) << 10) | uint(want.x));
      }}
    }}

    // March inside this resident brick: no more page-table traffic.
    vec3 poolBase = vec3(float(e & 0x3FFu), float((e >> 10) & 0x3FFu),
                         float((e >> 20) & 0x3FFu)) * borderedDim + 1.0;
    float stepRel = dt / dtMin;
    float tEnd = min(tExit, t1);

    // Sub-brick empty skipping: probe the covering coarser brick already in
    // the pool (each coarse voxel spans 8 samples here). A downsampled voxel
    // at/below the visibility cutoff is treated as authoritatively empty at
    // full res — a hair of accuracy at content boundaries traded for
    // skipping dead runs inside partially-empty bricks.
    int Lc = min(L + 3, kLevels - 1);
    ivec3 probeTexel = ivec3(0);
    ivec3 probeOrigin = ivec3(0);
    bool probeOk = false;
    if (Lc > L) {{
      // Brick spans at L are chunkDim-aligned, so one coarse brick covers it.
      ivec3 startLc = (brick * int(chunkDim)) >> (Lc - L);
      ivec3 bLc = startLc / int(chunkDim);
      uint ec = fetchPageEntry(Lc, bLc);
      if ((ec & 0x40000000u) != 0u) {{
        probeOk = true;
        probeTexel = ivec3(int(ec & 0x3FFu), int((ec >> 10) & 0x3FFu),
                           int((ec >> 20) & 0x3FFu)) * int(borderedDim) + 1;
        probeOrigin = bLc * int(chunkDim);
      }}
    }}

    int sRun = 0;
    for (int cell = 0; cell < 64 && t < tEnd && acc.a < 0.96; ++cell) {{
      float tCellEnd = tEnd;
      if (probeOk) {{
        ivec3 cv = ivec3(floor(((ro + t * rd) / volSizeWorld) *
                               levelShape[Lc]));
        vec3 wc = volSizeWorld / levelShape[Lc];
        vec3 cmin = vec3(cv) * wc;
        vec3 cfar = mix(cmin + wc, cmin, lessThan(rd, vec3(0.0)));
        float tCellExit = min(min((cfar.x - ro.x) * invRd.x,
                                  (cfar.y - ro.y) * invRd.y),
                              (cfar.z - ro.z) * invRd.z);
        ivec3 pv = clamp(cv - probeOrigin, ivec3(0), ivec3(int(chunkDim) - 1));
        float cd = texelFetch(brickPool, probeTexel + pv, 0).r;
        if (cd <= occSkipThreshold) {{
          t = max(tCellExit, t) + dt * 0.05;
          continue;
        }}
        tCellEnd = min(tCellExit + dt * 0.05, tEnd);
      }}
      // Incremental voxel-space stepping: within a run t advances by a
      // constant dt, so the position update is one add per sample instead
      // of the full world->voxel transform. Resynced on every skip/jump
      // because those rewrite t.
      vec3 vScaleL = levelShape[L] / volSizeWorld;
      vec3 vCur = (ro + t * rd) * vScaleL;
      vec3 vStep = (rd * dt) * vScaleL;
      vec3 brickBase = vec3(brick) * chunkDim;
      for (; sRun < 160 && t < tCellEnd && acc.a < 0.96; ++sRun) {{
        vec3 inBrick = clamp(vCur - brickBase, vec3(0.0), vec3(chunkDim));
        float dens = texture(brickPool, (poolBase + inBrick) * invPoolTexels).r;
        vec3 stencilGrad;
        dens = filteredDensity(dens, poolBase + inBrick, stencilGrad);
        float w = clamp((dens - winCenter) / winWidth + 0.5, 0.0, 1.0);
{3}        vec3 col = (renderMode == 0) ? vec3(w) : densRamp(w);
{6}{5}        if (renderMode == 2) {{
          // Opaque isosurface at winCenter: bisect the crossing inside this
          // brick, shade at the refined hit, and terminate the ray.
          if (dens >= winCenter && w > 0.001) {{
            float tA = max(t - dt, 0.0), tB = t;
            for (int bi = 0; bi < 5; ++bi) {{
              float tm = 0.5 * (tA + tB);
              vec3 vm = ((ro + tm * rd) / volSizeWorld) * levelShape[L];
              vec3 im = clamp(vm - vec3(brick) * chunkDim, vec3(0.0),
                              vec3(chunkDim));
              vec3 gTmp;
              if (filteredDensity(
                      texture(brickPool, (poolBase + im) * invPoolTexels).r,
                      poolBase + im, gTmp) >= winCenter)
                tB = tm;
              else
                tA = tm;
            }}
            vec3 vh = ((ro + tB * rd) / volSizeWorld) * levelShape[L];
            vec3 ih = clamp(vh - vec3(brick) * chunkDim, vec3(0.0),
                            vec3(chunkDim));
            vec3 g = poolGrad(poolBase + ih);
            vec3 N = (dot(g, g) > 1e-10) ? -normalize(g) : -rd;
            if (dot(N, rd) > 0.0) N = -N;
            float sh = shadowFactor(ro + tB * rd, L);
            acc.rgb += (1.0 - acc.a) * shadeSurface(col, N, rd, sh);
            acc.a = 1.0;
          }}
          t += dt;
          vCur += vStep;
          continue;
        }}
        if (renderMode == 1 && w > 0.02) {{
          // Gradient-magnitude-modulated lighting: homogeneous interiors
          // stay unshaded, boundaries pick up the surface model. The filter
          // stencil already produced the gradient; fetch only if it didn't.
          vec3 g = (filterOp != 0) ? stencilGrad
                                   : poolGrad(poolBase + inBrick);
          float gm = length(g);
          if (gm > 1e-4) {{
            vec3 N = -g / gm;
            if (dot(N, rd) > 0.0) N = -N;
            col = mix(col, shadeSurface(col, N, rd, 1.0),
                      clamp(gm * 40.0, 0.0, 1.0));
          }}
        }}
        // Opacity-adaptive step stretching: behind accumulated opacity the
        // remaining contribution shrinks, so stride up to 3x with the
        // compositing corrected by the effective step length. Cuts the
        // saturation tail of deep-zoom rays roughly in half.
        float stretch = 1.0 + acc.a * 2.0;
        // 1-(1-t)^s via exp2: ln(1-t) ~ -(t + t^2/2) (t <= ~0.25 under the
        // opacity slider; error < 1%), saving a log per sample vs pow().
        float tt = w * opacityScale;
        float a = 1.0 - exp2(-1.442695 * stepRel * stretch *
                             (tt + 0.5 * tt * tt));
        acc.rgb += (1.0 - acc.a) * a * col;
        acc.a   += (1.0 - acc.a) * a;
        t += dt * stretch;
        vCur += vStep * stretch;
      }}
      if (sRun >= 160) break;
    }}
    // The final t already sits < dt into the next brick, so sampling stays
    // continuous across the boundary. Nudge only on a zero-sample stall
    // (float-precision entry exactly on a face).
    if (sRun == 0) t = max(t, tExit) + dt * 0.05;
  }}
  if (acc.a < 0.004) discard;
  // Terminated at 0.96: treat as fully saturated to avoid a haze band.
  if (acc.a > 0.955) acc /= acc.a;
  fragColor = acc;
}}
)glsl",
                     BuildCommonGlsl(), maskDecls,
                     hasMask ? "    if (maskIsolate > 0.5 &&\n"
                               "        maskPresent(L, brick) < 0.5) {\n"
                               "      t = max(tExit, t) + dt * 0.05;\n"
                               "      continue;\n"
                               "    }\n"
                             : "",
                     maskWeight, ovDecls, ovSample, maskTint);
}

std::string vtkScrollVolumeMapper::BuildSurfaceVertexShader() const {
  return R"glsl(//VTK::System::Dec
in vec3 inPos;
in vec3 inNrm;
uniform mat4 viewProj;
out vec3 wPos;
out vec3 wNrm;
void main() {
  wPos = inPos;
  wNrm = inNrm;
  gl_Position = viewProj * vec4(inPos, 1.0);
}
)glsl";
}

std::string vtkScrollVolumeMapper::BuildSurfaceFragmentShader() const {
  return std::format(R"glsl(//VTK::System::Dec
#extension GL_ARB_shader_storage_buffer_object : require
in vec3 wPos;
in vec3 wNrm;
out vec4 fragColor;
{0}
uniform vec3 camPos;
uniform float slabFront;   // native voxels along -normal
uniform float slabBehind;  // native voxels along +normal
uniform int slabMean;      // 0 = max over slab, 1 = mean

// Single-point sample through the page tables with coarse fallback.
// usedL reports the level actually sampled (kLevels = nothing resident).
float sampleAt(vec3 pw, int lod, out int usedL) {{
  vec3 pn = pw / volSizeWorld;
  if (any(lessThan(pn, vec3(0.0))) || any(greaterThanEqual(pn, vec3(1.0)))) {{
    usedL = lod;
    return 0.0;
  }}
  for (int i = lod; i < kLevels; ++i) {{
    vec3 v = pn * levelShape[i];
    ivec3 b = ivec3(clamp(floor(v / chunkDim), vec3(0.0), gridDim[i] - 1.0));
    uint pe = fetchPageEntry(i, b);
    if ((pe & 0x80000000u) != 0u) {{  // known empty: fill value
      usedL = i;
      return 0.0;
    }}
    if ((pe & 0x40000000u) != 0u) {{
      usedL = i;
      vec3 poolBase = vec3(float(pe & 0x3FFu), float((pe >> 10) & 0x3FFu),
                           float((pe >> 20) & 0x3FFu)) * borderedDim + 1.0;
      vec3 inBrick = clamp(v - vec3(b) * chunkDim, vec3(0.0), vec3(chunkDim));
      return texture(brickPool, (poolBase + inBrick) * invPoolTexels).r;
    }}
  }}
  usedL = kLevels;
  return 0.0;
}}

void main() {{
  int minLod = int(clamp(desiredLevel, 0.0, float(kLevels - 1)));
  float wpp = wppConst + wppPerDist * distance(camPos, wPos);
  int lod = clamp(int(floor(log2(max(1.0, wpp / voxelWorld0)))), minLod,
                  kLevels - 1);
  vec3 n = normalize(wNrm);

  // Slab is defined in native voxels; sample at the chosen level's step so
  // deep-zoom-out does not pay one texture tap per native voxel.
  float lodScale = exp2(float(lod));
  float dt = voxelWorld0 * lodScale;
  int nf = min(int(ceil(slabFront / lodScale)), 64);
  int nb = min(int(ceil(slabBehind / lodScale)), 64);

  float acc = 0.0;
  float cnt = 0.0;
  int coarsest = -1;
  for (int s = -nf; s <= nb; ++s) {{
    int usedL;
    float d = sampleAt(wPos + n * (float(s) * dt), lod, usedL);
    coarsest = max(coarsest, usedL);
    acc = (slabMean == 1) ? acc + d : max(acc, d);
    cnt += 1.0;
  }}
  if (slabMean == 1) acc /= max(cnt, 1.0);

  // Sampled coarser than wanted anywhere in the slab: request the finer
  // brick under the surface point (same feedback stream as the raymarcher).
  if (coarsest > lod) {{
    vec3 vw = (wPos / volSizeWorld) * levelShape[lod];
    ivec3 want = ivec3(clamp(floor(vw / chunkDim), vec3(0.0),
                             gridDim[lod] - 1.0));
    uint idx = atomicAdd(reqCount, 1u);
    if (idx < kMaxReq)
      reqs[idx] = uvec2(uint(lod), (uint(want.z) << 20) |
                                   (uint(want.y) << 10) | uint(want.x));
  }}

  float w = clamp((acc - winCenter) / winWidth + 0.5, 0.0, 1.0);
  fragColor = vec4(vec3(w), 1.0);
}}
)glsl",
                     BuildCommonGlsl());
}

void vtkScrollVolumeMapper::EnsureSurfaceMesh() {
  if (!SurfaceDirty) return;
  SurfaceDirty = false;
  SurfIndexCount = 0;
  if (!Surface) return;

  const auto& pts = Surface->points();
  const std::uint32_t W = Surface->width(), H = Surface->height();
  const float sp = float(Spacing) * SurfScale;

  // Interleaved position + normal; normals from central differences of the
  // parametric grid (valid neighbors only, one-sided at edges/holes).
  std::vector<float> vb(std::size_t{W} * H * 6, 0.f);
  auto P = [&](std::uint32_t u, std::uint32_t v) {
    return pts[std::size_t{v} * W + u];
  };
  auto ok = [&](std::uint32_t u, std::uint32_t v) {
    return Surface->valid(u, v);
  };
  for (std::uint32_t v = 0; v < H; ++v)
    for (std::uint32_t u = 0; u < W; ++u) {
      float* out = vb.data() + (std::size_t{v} * W + u) * 6;
      if (!ok(u, v)) continue;
      const auto p = P(u, v);
      out[0] = p[0] * sp;
      out[1] = p[1] * sp;
      out[2] = p[2] * sp;
      auto diff = [&](int du, int dv, int axis) -> float {
        const bool hasP = u + du < W && v + dv < H && ok(u + du, v + dv);
        const bool hasM =
            int(u) - du >= 0 && int(v) - dv >= 0 && ok(u - du, v - dv);
        const auto& c = P(u, v);
        if (hasP && hasM) return P(u + du, v + dv)[axis] - P(u - du, v - dv)[axis];
        if (hasP) return 2.f * (P(u + du, v + dv)[axis] - c[axis]);
        if (hasM) return 2.f * (c[axis] - P(u - du, v - dv)[axis]);
        return 0.f;
      };
      const float du[3] = {diff(1, 0, 0), diff(1, 0, 1), diff(1, 0, 2)};
      const float dv[3] = {diff(0, 1, 0), diff(0, 1, 1), diff(0, 1, 2)};
      float nx = du[1] * dv[2] - du[2] * dv[1];
      float ny = du[2] * dv[0] - du[0] * dv[2];
      float nz = du[0] * dv[1] - du[1] * dv[0];
      const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
      if (len > 0.f) {
        out[3] = nx / len;
        out[4] = ny / len;
        out[5] = nz / len;
      }
    }

  std::vector<std::uint32_t> ib;
  ib.reserve(std::size_t{W} * H * 6);
  for (std::uint32_t v = 0; v + 1 < H; ++v)
    for (std::uint32_t u = 0; u + 1 < W; ++u) {
      if (!ok(u, v) || !ok(u + 1, v) || !ok(u, v + 1) || !ok(u + 1, v + 1))
        continue;
      const std::uint32_t i00 = v * W + u, i10 = v * W + u + 1;
      const std::uint32_t i01 = (v + 1) * W + u, i11 = (v + 1) * W + u + 1;
      ib.insert(ib.end(), {i00, i10, i11, i00, i11, i01});
    }

  if (!SurfVao) {
    glGenVertexArrays(1, &SurfVao);
    glGenBuffers(1, &SurfVbo);
    glGenBuffers(1, &SurfIbo);
  }
  glBindVertexArray(SurfVao);
  glBindBuffer(GL_ARRAY_BUFFER, SurfVbo);
  glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(vb.size() * sizeof(float)),
               vb.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, SurfIbo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER,
               GLsizeiptr(ib.size() * sizeof(std::uint32_t)), ib.data(),
               GL_STATIC_DRAW);
  glBindVertexArray(0);
  SurfIndexCount = int(ib.size());
  logInfo("surface mesh: {}x{} grid, {} triangles", W, H,
          SurfIndexCount / 3);
}

void vtkScrollVolumeMapper::RenderSurface(vtkOpenGLRenderWindow* renWin,
                                          vtkRenderer* ren) {
  if (!Surface || !ShowSurface) return;
  EnsureSurfaceMesh();
  if (!SurfIndexCount) return;

  if (SurfVsSrc.empty()) {
    SurfVsSrc = BuildSurfaceVertexShader();
    SurfFsSrc = BuildSurfaceFragmentShader();
  }
  vtkShaderProgram* prog = renWin->GetShaderCache()->ReadyShaderProgram(
      SurfVsSrc.c_str(), SurfFsSrc.c_str(), "");
  if (!prog || !prog->GetCompiled()) {
    if (FrameIndex < 3) logError("surface shader failed to compile");
    return;
  }

  // Shared textures were Activated by Render(); rebind their units here.
  prog->SetUniformi("brickPool", Cache->poolTexture()->GetTextureUnit());
  for (int i = 0; i < Cache->levelCount(); ++i)
    prog->SetUniformi(std::format("pageDir{}", i).c_str(),
                      Cache->directoryTexture(i)->GetTextureUnit());
  prog->SetUniformi("tilePool", Cache->tilePoolTexture()->GetTextureUnit());
  prog->SetUniformi("occPool", Cache->occPoolTexture()->GetTextureUnit());
  const auto tpd = Cache->tilePoolDims();
  prog->SetUniformi("tilePoolDimX", int(tpd[2]));
  prog->SetUniformi("tilePoolDimY", int(tpd[1]));

  const float volSize[3] = {float(double(ShapeZyx[2]) * Spacing),
                            float(double(ShapeZyx[1]) * Spacing),
                            float(double(ShapeZyx[0]) * Spacing)};
  prog->SetUniform3f("volSizeWorld", volSize);
  const int levels = Cache->levelCount();
  std::vector<float> levelShape(levels * 3), gridDim(levels * 3);
  for (int i = 0; i < levels; ++i) {
    const auto g = Cache->levelDims(i).grid;
    for (int a = 0; a < 3; ++a) {
      const std::uint64_t full = ShapeZyx[2 - a];
      levelShape[i * 3 + a] =
          float(std::max<std::uint64_t>(1, (full + ((1ull << i) - 1)) >> i));
    }
    gridDim[i * 3 + 0] = float(g[2]);
    gridDim[i * 3 + 1] = float(g[1]);
    gridDim[i * 3 + 2] = float(g[0]);
  }
  prog->SetUniform3fv("levelShape", levels,
                      reinterpret_cast<const float(*)[3]>(levelShape.data()));
  prog->SetUniform3fv("gridDim", levels,
                      reinterpret_cast<const float(*)[3]>(gridDim.data()));
  const auto slotDims = Cache->poolSlotDims();
  const float bd = float(Cache->borderedDim());
  const float poolTexels[3] = {slotDims[2] * bd, slotDims[1] * bd,
                               slotDims[0] * bd};
  prog->SetUniform3f("poolTexels", poolTexels);
  const float invPoolTexels[3] = {1.f / poolTexels[0], 1.f / poolTexels[1],
                                  1.f / poolTexels[2]};
  prog->SetUniform3f("invPoolTexels", invPoolTexels);
  prog->SetUniformf("chunkDim", float(ChunkDim));
  prog->SetUniformf("borderedDim", bd);
  prog->SetUniformf("desiredLevel", DesiredLevel);
  prog->SetUniformf("voxelWorld0", float(Spacing));
  prog->SetUniformf("winWidth", Window);
  prog->SetUniformf("winCenter", Level);
  prog->SetUniformf("slabFront", SlabFront);
  prog->SetUniformf("slabBehind", SlabBehind);
  prog->SetUniformi("slabMean", SlabMean ? 1 : 0);

  vtkCamera* cam = ren->GetActiveCamera();
  vtkMatrix4x4* wcdc = cam->GetCompositeProjectionTransformMatrix(
      ren->GetTiledAspectRatio(), -1, 1);
  vtkNew<vtkMatrix4x4> vp;
  vp->DeepCopy(wcdc);
  vp->Transpose();
  prog->SetUniformMatrix("viewProj", vp);
  const double* cp = cam->GetPosition();
  const float camPos[3] = {float(cp[0]), float(cp[1]), float(cp[2])};
  prog->SetUniform3f("camPos", camPos);

  const int vpHeight = std::max(1, ren->GetSize()[1]);
  float wppC = 0.f, wppD = 0.f;
  if (cam->GetParallelProjection()) {
    wppC = float(2.0 * cam->GetParallelScale() / vpHeight);
  } else {
    wppD = float(
        2.0 * std::tan(vtkMath::RadiansFromDegrees(cam->GetViewAngle()) / 2) /
        vpHeight);
  }
  prog->SetUniformf("wppConst", wppC);
  prog->SetUniformf("wppPerDist", wppD);
  prog->SetUniformf("sampleStepScale", SampleStepScale);
  prog->SetUniformf("opacityScale", OpacityScale);
  prog->SetUniformf("occSkipThreshold", std::max(0.f, Level - 0.5f * Window));

  vtkOpenGLState* state = renWin->GetState();
  state->vtkglEnable(GL_DEPTH_TEST);
  state->vtkglDepthMask(GL_TRUE);
  state->vtkglDisable(GL_BLEND);
  glDisable(GL_CULL_FACE);

  glBindVertexArray(SurfVao);
  glBindBuffer(GL_ARRAY_BUFFER, SurfVbo);
  const GLuint handle = static_cast<GLuint>(prog->GetHandle());
  const GLint locPos = glGetAttribLocation(handle, "inPos");
  const GLint locNrm = glGetAttribLocation(handle, "inNrm");
  if (locPos >= 0) {
    glEnableVertexAttribArray(GLuint(locPos));
    glVertexAttribPointer(GLuint(locPos), 3, GL_FLOAT, GL_FALSE,
                          6 * sizeof(float), nullptr);
  }
  if (locNrm >= 0) {
    glEnableVertexAttribArray(GLuint(locNrm));
    glVertexAttribPointer(GLuint(locNrm), 3, GL_FLOAT, GL_FALSE,
                          6 * sizeof(float),
                          reinterpret_cast<void*>(3 * sizeof(float)));
  }
  glDrawElements(GL_TRIANGLES, SurfIndexCount, GL_UNSIGNED_INT, nullptr);
  glBindVertexArray(0);
}

void vtkScrollVolumeMapper::RenderClusters(vtkOpenGLRenderWindow* renWin,
                                           vtkRenderer* ren) {
  if (Clusters.empty()) return;

  if (ClusterVsSrc.empty()) {
    ClusterVsSrc = R"glsl(//VTK::System::Dec
in vec3 inPos;
in float inRad;
in float inVal;
uniform mat4 viewProj;
uniform float pointScale;   // pixels per world unit at w == 1
uniform float pointConst;   // fixed pixels per world unit (ortho), or 0
out float vVal;
void main() {
  gl_Position = viewProj * vec4(inPos, 1.0);
  float r = inRad * 1.6;  // inflate: neighboring splats overlap and fuse
  float px = (pointConst > 0.0)
                 ? pointConst * r
                 : pointScale * r / max(gl_Position.w, 1e-4);
  gl_PointSize = clamp(2.0 * px, 1.5, 1024.0);
  vVal = inVal;
}
)glsl";
    ClusterFsSrc = R"glsl(//VTK::System::Dec
in float vVal;
out vec4 fragColor;
uniform vec3 cluLightDir;   // view-space key light
uniform float cluAmbient;
void main() {
  vec2 pc = gl_PointCoord * 2.0 - 1.0;
  pc.y = -pc.y;
  float r2 = dot(pc, pc);
  if (r2 > 1.0) discard;
  // Gaussian falloff: soft splats fuse where clusters overlap.
  float alpha = exp(-3.5 * r2);

  vec3 N = vec3(pc, sqrt(1.0 - r2));
  vec3 Ld = normalize(cluLightDir);
  float ndl = max(dot(N, Ld), 0.0);
  // Viridis (Zucker's polynomial fit), t = normalized cluster mean.
  float t = clamp(vVal, 0.0, 1.0);
  vec3 alb = vec3(0.2777273272234177, 0.005407344544966578,
                  0.3340998053353061) +
             t * (vec3(0.1050930431085774, 1.404613529898575,
                       1.384590162594685) +
             t * (vec3(-0.3308618287255563, 0.214847559468213,
                       0.09509516302823659) +
             t * (vec3(-4.634230498983486, -5.799100973351585,
                       -19.33244095627987) +
             t * (vec3(6.228269936347081, 14.17993336680509,
                       56.69055260068105) +
             t * (vec3(4.776384997670288, -13.74514537774601,
                       -65.35303263337234) +
             t * vec3(-5.435455855934631, 4.645852612178535,
                      26.3124352495832))))));
  alb = clamp(alb, 0.0, 1.0);
  vec3 H = normalize(Ld + vec3(0.0, 0.0, 1.0));
  float spec = pow(max(dot(N, H), 0.0), 48.0);
  float fres = pow(1.0 - max(N.z, 0.0), 3.0);   // rim
  float fill = max(dot(N, -Ld), 0.0);
  vec3 col = alb * (cluAmbient * (0.20 + 0.06 * N.y) + 0.16 * fill * cluAmbient
                    + 0.85 * ndl) +
             vec3(spec * 0.40 + fres * 0.06);
  fragColor = vec4(col * alpha, alpha);  // premultiplied
}
)glsl";
  }

  vtkShaderProgram* prog = renWin->GetShaderCache()->ReadyShaderProgram(
      ClusterVsSrc.c_str(), ClusterFsSrc.c_str(), "");
  if (!prog || !prog->GetCompiled()) {
    if (FrameIndex < 3) logError("cluster shader failed to compile");
    return;
  }

  if (!ClusterVao) {
    glGenVertexArrays(1, &ClusterVao);
    glGenBuffers(1, &ClusterVbo);
  }
  glBindVertexArray(ClusterVao);
  glBindBuffer(GL_ARRAY_BUFFER, ClusterVbo);
  {
    // Blended splats need back-to-front order. Re-sort + re-upload only
    // when the view direction actually moved (> ~0.6 deg) or the cluster
    // set changed; a static camera costs nothing.
    vtkCamera* scam = ren->GetActiveCamera();
    const double* dop = scam->GetDirectionOfProjection();
    const double dot = dop[0] * LastSortDop[0] + dop[1] * LastSortDop[1] +
                       dop[2] * LastSortDop[2];
    if (ClustersDirty || dot < 0.99995) {
      LastSortDop[0] = dop[0];
      LastSortDop[1] = dop[1];
      LastSortDop[2] = dop[2];
      ClusterSorted.assign(Clusters.begin(), Clusters.end());
      std::ranges::sort(ClusterSorted, [&](const ClusterSphere& a,
                                           const ClusterSphere& b) {
        const double da = a.x * dop[0] + a.y * dop[1] + a.z * dop[2];
        const double db = b.x * dop[0] + b.y * dop[1] + b.z * dop[2];
        return da > db;  // farthest along view direction first
      });
      glBufferData(GL_ARRAY_BUFFER,
                   GLsizeiptr(ClusterSorted.size() * sizeof(ClusterSphere)),
                   ClusterSorted.data(), GL_STREAM_DRAW);
      ClustersDirty = false;
    }
  }

  vtkCamera* cam = ren->GetActiveCamera();
  vtkMatrix4x4* wcdc = cam->GetCompositeProjectionTransformMatrix(
      ren->GetTiledAspectRatio(), -1, 1);
  vtkNew<vtkMatrix4x4> vp;
  vp->DeepCopy(wcdc);
  // vtkShaderProgram::SetUniformMatrix uploads vtkMatrix4x4 row-major with
  // transpose=GL_FALSE, so GLSL sees the transpose; pre-transpose so
  // `viewProj * v` is the actual world->DC transform. (Verified empirically
  // with the synthetic cluster pattern.)
  vp->Transpose();
  prog->SetUniformMatrix("viewProj", vp);


  const int vpHeight = std::max(1, ren->GetSize()[1]);
  float scale = 0.f, cst = 0.f;
  if (cam->GetParallelProjection())
    cst = float(vpHeight / (2.0 * cam->GetParallelScale()));
  else
    scale = float(
        vpHeight /
        (2.0 *
         std::tan(vtkMath::RadiansFromDegrees(cam->GetViewAngle()) / 2)));
  prog->SetUniformf("pointScale", scale);
  prog->SetUniformf("pointConst", cst);

  // Impostor normals are view-space: rotate the world key light into view
  // space (rotation part of the view transform only).
  vtkMatrix4x4* vm = cam->GetViewTransformMatrix();
  float lv[3];
  for (int i = 0; i < 3; ++i)
    lv[i] = float(vm->GetElement(i, 0)) * LightDir[0] +
            float(vm->GetElement(i, 1)) * LightDir[1] +
            float(vm->GetElement(i, 2)) * LightDir[2];
  prog->SetUniform3f("cluLightDir", lv);
  prog->SetUniformf("cluAmbient", LightAmbient);

  vtkOpenGLState* state = renWin->GetState();
  state->vtkglEnable(GL_DEPTH_TEST);
  state->vtkglDepthMask(GL_FALSE);  // blended splats: test, don't write
  state->vtkglEnable(GL_BLEND);
  state->vtkglBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
                                GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_PROGRAM_POINT_SIZE);

  const GLuint handle = static_cast<GLuint>(prog->GetHandle());
  const GLint locPos = glGetAttribLocation(handle, "inPos");
  const GLint locRad = glGetAttribLocation(handle, "inRad");
  const GLint locVal = glGetAttribLocation(handle, "inVal");
  const GLsizei stride = sizeof(ClusterSphere);
  if (locPos >= 0) {
    glEnableVertexAttribArray(GLuint(locPos));
    glVertexAttribPointer(GLuint(locPos), 3, GL_FLOAT, GL_FALSE, stride,
                          nullptr);
  }
  if (locRad >= 0) {
    glEnableVertexAttribArray(GLuint(locRad));
    glVertexAttribPointer(GLuint(locRad), 1, GL_FLOAT, GL_FALSE, stride,
                          reinterpret_cast<void*>(3 * sizeof(float)));
  }
  if (locVal >= 0) {
    glEnableVertexAttribArray(GLuint(locVal));
    glVertexAttribPointer(GLuint(locVal), 1, GL_FLOAT, GL_FALSE, stride,
                          reinterpret_cast<void*>(4 * sizeof(float)));
  }
  glDrawArrays(GL_POINTS, 0, GLsizei(ClusterSorted.size()));
  glBindVertexArray(0);
  glDisable(GL_PROGRAM_POINT_SIZE);
  state->vtkglDepthMask(GL_TRUE);
}

void vtkScrollVolumeMapper::Render(vtkRenderer* ren, vtkVolume*) {
  auto* renWin = static_cast<vtkOpenGLRenderWindow*>(ren->GetRenderWindow());
  if (FrameIndex == 0) logInfo("mapper Render entered (first frame)");
  if (!Cache) return;
  ProfScope timeFrame(profStages().frame);

  // Harvest last frame's GPU raymarch time if the query result landed.
  if (GpuTimerInFlight) {
    GLuint available = 0;
    glGetQueryObjectuiv(GpuTimerQuery, GL_QUERY_RESULT_AVAILABLE, &available);
    if (available) {
      GLuint64 ns = 0;
      glGetQueryObjectui64v(GpuTimerQuery, GL_QUERY_RESULT, &ns);
      profStages().gpuRaymarch.add(ns);
      GpuTimerInFlight = false;
    }
  }

  if (!Cache->initialized()) {
    if (!Primary) return;  // secondary windows wait for the primary
    Cache->initialize(renWin);
  }
  ++FrameIndex;
  if (Primary) {
    Cache->frameBegin(FrameIndex);
    if (PreRender) PreRender();  // drain pipeline, upload (context current)
    Cache->syncPageTables();
  }

  if (!Quad) {
    const std::string fs = BuildFragmentShader();
    Quad = std::make_unique<vtkOpenGLQuadHelper>(renWin, nullptr, fs.c_str(),
                                                 nullptr);
    if (!Quad->Program || !Quad->Program->GetCompiled()) {
      logError("raymarch shader failed to compile");
      Quad.reset();
      return;
    }
  }

  renWin->GetShaderCache()->ReadyShaderProgram(Quad->Program);
  vtkShaderProgram* prog = Quad->Program;

  // Textures: pool on unit 0, page tables on 1..N.
  Cache->poolTexture()->Activate();
  if (NearestTaps != AppliedNearest) {
    // Flip the pool's sampler state in place; every shader tap follows.
    const GLint f = NearestTaps ? GL_NEAREST : GL_LINEAR;
    glActiveTexture(GL_TEXTURE0 + Cache->poolTexture()->GetTextureUnit());
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, f);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, f);
    AppliedNearest = NearestTaps;
  }
  prog->SetUniformi("brickPool",
                    Cache->poolTexture()->GetTextureUnit());
  for (int i = 0; i < Cache->levelCount(); ++i) {
    auto* dir = Cache->directoryTexture(i);
    dir->Activate();
    prog->SetUniformi(std::format("pageDir{}", i).c_str(),
                      dir->GetTextureUnit());
  }
  Cache->tilePoolTexture()->Activate();
  prog->SetUniformi("tilePool", Cache->tilePoolTexture()->GetTextureUnit());
  Cache->occPoolTexture()->Activate();
  prog->SetUniformi("occPool", Cache->occPoolTexture()->GetTextureUnit());
  const auto tpd = Cache->tilePoolDims();
  prog->SetUniformi("tilePoolDimX", static_cast<int>(tpd[2]));
  prog->SetUniformi("tilePoolDimY", static_cast<int>(tpd[1]));

  if (Mask && Mask->uploaded()) {
    const int dirUnit = 26, poolUnit = 27;  // clear of VTK's units
    glActiveTexture(GL_TEXTURE0 + dirUnit);
    glBindTexture(GL_TEXTURE_3D, Mask->directoryTexture());
    glActiveTexture(GL_TEXTURE0 + poolUnit);
    glBindTexture(GL_TEXTURE_3D, Mask->blockPoolTexture());
    prog->SetUniformi("maskDir", dirUnit);
    prog->SetUniformi("maskPool", poolUnit);
    const auto mpd = Mask->poolDims();
    prog->SetUniformi("maskPoolDimX", int(mpd[2]));
    prog->SetUniformi("maskPoolDimY", int(mpd[1]));
    prog->SetUniformf("maskStrength", MaskStrength);
    prog->SetUniformf("maskIsolate", MaskIsolate ? 1.f : 0.f);
    // Presence mip on fixed units 18..23 (above VTK's ~17 managed units).
    const int nPres = std::min(Cache->levelCount(),
                               Mask->presenceLevelCount());
    for (int i = 0; i < nPres; ++i) {
      const int u = 18 + i;
      glActiveTexture(GL_TEXTURE0 + u);
      glBindTexture(GL_TEXTURE_3D, Mask->presenceTexture(i));
      prog->SetUniformi(std::format("maskLvl{}", i).c_str(), u);
    }
  }

  if (Overlay && Overlay->initialized()) {
    Overlay->poolTexture()->Activate();
    prog->SetUniformi("ovBrickPool",
                      Overlay->poolTexture()->GetTextureUnit());
    for (int i = 0; i < Overlay->levelCount(); ++i) {
      auto* dir = Overlay->directoryTexture(i);
      dir->Activate();
      prog->SetUniformi(std::format("ovPageDir{}", i).c_str(),
                        dir->GetTextureUnit());
    }
    Overlay->tilePoolTexture()->Activate();
    prog->SetUniformi("ovTilePool",
                      Overlay->tilePoolTexture()->GetTextureUnit());
    const auto otpd = Overlay->tilePoolDims();
    prog->SetUniformi("ovTilePoolDimX", int(otpd[2]));
    prog->SetUniformi("ovTilePoolDimY", int(otpd[1]));

    const int ovLevels = Overlay->levelCount();
    std::vector<float> ovShape(ovLevels * 3), ovGrid(ovLevels * 3);
    for (int i = 0; i < ovLevels; ++i) {
      const auto g = Overlay->levelDims(i).grid;
      for (int a = 0; a < 3; ++a) {
        const std::uint64_t full = OverlayShapeZyx[2 - a];
        ovShape[i * 3 + a] =
            float(std::max<std::uint64_t>(1, (full + ((1ull << i) - 1)) >> i));
      }
      ovGrid[i * 3 + 0] = float(g[2]);
      ovGrid[i * 3 + 1] = float(g[1]);
      ovGrid[i * 3 + 2] = float(g[0]);
    }
    prog->SetUniform3fv("ovLevelShape", ovLevels,
                        reinterpret_cast<const float(*)[3]>(ovShape.data()));
    prog->SetUniform3fv("ovGridDim", ovLevels,
                        reinterpret_cast<const float(*)[3]>(ovGrid.data()));
    const auto oSlots = Overlay->poolSlotDims();
    const float obd = float(Overlay->borderedDim());
    const float oPool[3] = {oSlots[2] * obd, oSlots[1] * obd,
                            oSlots[0] * obd};
    prog->SetUniform3f("ovPoolTexels", oPool);
    prog->SetUniformf("ovChunkDim", obd - 2.f);
    prog->SetUniformf("ovBorderedDim", obd);
    // lod shift = log2(main L0 voxel count / overlay L0 voxel count).
    int shift = 0;
    while ((OverlayShapeZyx[0] << (shift + 1)) <= ShapeZyx[0]) ++shift;
    prog->SetUniformi("ovLevelOffset", shift);
    prog->SetUniformf("ovStrength", OverlayStrength);
    prog->SetUniformf("ovThreshold", OverlayThreshold);
  }

  // Camera: world -> DC, inverted. SetUniformMatrix uploads vtkMatrix4x4
  // row-major with transpose=false, so the matrix must be transposed on the
  // CPU for GLSL's column-vector convention (VTK's own camera matrices get
  // the same treatment in vtkOpenGLCamera::GetKeyMatrices).
  vtkCamera* cam = ren->GetActiveCamera();
  vtkMatrix4x4* wcdc = cam->GetCompositeProjectionTransformMatrix(
      ren->GetTiledAspectRatio(), -1, 1);
  vtkNew<vtkMatrix4x4> inv;
  vtkMatrix4x4::Invert(wcdc, inv);
  inv->Transpose();
  prog->SetUniformMatrix("invViewProj", inv);

  const float volSize[3] = {float(double(ShapeZyx[2]) * Spacing),
                            float(double(ShapeZyx[1]) * Spacing),
                            float(double(ShapeZyx[0]) * Spacing)};
  prog->SetUniform3f("volSizeWorld", volSize);

  const int levels = Cache->levelCount();
  std::vector<float> levelShape(levels * 3), gridDim(levels * 3);
  for (int i = 0; i < levels; ++i) {
    const auto g = Cache->levelDims(i).grid;
    // Level shape derived from full-res shape halved per level (x,y,z order).
    for (int a = 0; a < 3; ++a) {
      const std::uint64_t full = ShapeZyx[2 - a];
      levelShape[i * 3 + a] =
          float(std::max<std::uint64_t>(1, (full + ((1ull << i) - 1)) >> i));
    }
    gridDim[i * 3 + 0] = float(g[2]);
    gridDim[i * 3 + 1] = float(g[1]);
    gridDim[i * 3 + 2] = float(g[0]);
  }
  prog->SetUniform3fv("levelShape", levels,
                      reinterpret_cast<const float(*)[3]>(levelShape.data()));
  prog->SetUniform3fv("gridDim", levels,
                      reinterpret_cast<const float(*)[3]>(gridDim.data()));

  const auto slotDims = Cache->poolSlotDims();
  const float bd = float(Cache->borderedDim());
  const float poolTexels[3] = {slotDims[2] * bd, slotDims[1] * bd,
                               slotDims[0] * bd};
  prog->SetUniform3f("poolTexels", poolTexels);
  const float invPoolTexels[3] = {1.f / poolTexels[0], 1.f / poolTexels[1],
                                  1.f / poolTexels[2]};
  prog->SetUniform3f("invPoolTexels", invPoolTexels);
  prog->SetUniformf("chunkDim", float(ChunkDim));
  prog->SetUniformf("borderedDim", bd);
  prog->SetUniformf("desiredLevel", DesiredLevel);
  prog->SetUniformf("sampleStepScale", SampleStepScale);

  // Pixel footprint for cone LOD: world units per pixel, split into a
  // constant (ortho) and per-distance (perspective) part. Under reduced-res
  // interaction rendering the effective pixel grows, coarsening LOD too.
  const int vpHeight =
      std::max(1, int(float(ren->GetSize()[1]) * RenderScale));
  float wppC = 0.f, wppD = 0.f;
  if (cam->GetParallelProjection()) {
    wppC = float(2.0 * cam->GetParallelScale() / vpHeight);
  } else {
    wppD = float(2.0 *
                 std::tan(vtkMath::RadiansFromDegrees(cam->GetViewAngle()) / 2) /
                 vpHeight);
  }
  prog->SetUniformf("wppConst", wppC);
  prog->SetUniformf("wppPerDist", wppD);
  prog->SetUniformf("voxelWorld0", float(Spacing));
  prog->SetUniformf("winWidth", Window);
  prog->SetUniformf("winCenter", Level);
  prog->SetUniformi("renderMode", RenderMode);
  prog->SetUniformi("filterOp", FilterOp);
  prog->SetUniformf("filterAmt", FilterAmount);
  prog->SetUniformf("filterFloor", FilterFloor);
  prog->SetUniform3f("lightDir", LightDir);
  prog->SetUniformf("lightAmbient", LightAmbient);
  prog->SetUniformi("shadowsOn", ShadowsOn ? 1 : 0);
  prog->SetUniformf("opacityScale", OpacityScale);
  // Bricks whose max density maps to zero opacity under the current window
  // are culled exactly like empty ones. The high-pass floor culls the same
  // way — except in edge mode, where it thresholds the gradient field, not
  // raw density.
  prog->SetUniformf("occSkipThreshold",
                    std::max({0.f, Level - 0.5f * Window,
                              FilterOp == 3 ? 0.f : FilterFloor}));

  if (!FeedbackSsbo[0]) InitFeedbackBuffers();
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0,
                   FeedbackSsbo[FeedbackWriteIndex]);

  vtkOpenGLState* state = renWin->GetState();
  state->vtkglDepthMask(GL_FALSE);
  state->vtkglDisable(GL_DEPTH_TEST);

  const bool timeThisFrame = !GpuTimerInFlight;
  if (timeThisFrame) {
    if (!GpuTimerQuery) glGenQueries(1, &GpuTimerQuery);
    glBeginQuery(GL_TIME_ELAPSED, GpuTimerQuery);
  }

  const bool clusterView = RenderMode == 3 && !Clusters.empty();
  const bool lowRes = ShowVolume && !clusterView && RenderScale < 0.999f;
  if (!ShowVolume || clusterView) {
    // Surface-only / cluster view: nothing to raymarch or composite.
  } else if (lowRes) {
    // Raymarch into a reduced-res target, then composite upscaled.
    int rw = 0, rh = 0, rx = 0, ry = 0;
    ren->GetTiledSizeAndOrigin(&rw, &rh, &rx, &ry);
    const int sw = std::max(1, int(float(rw) * RenderScale));
    const int sh = std::max(1, int(float(rh) * RenderScale));
    EnsureLowResTarget(sw, sh);

    state->PushFramebufferBindings();
    glBindFramebuffer(GL_FRAMEBUFFER, LowResFbo);
    state->vtkglViewport(0, 0, sw, sh);
    state->vtkglDisable(GL_BLEND);
    glClearColor(0.f, 0.f, 0.f, 0.f);
    glClear(GL_COLOR_BUFFER_BIT);

    Quad->Render();

    state->PopFramebufferBindings();
    state->vtkglViewport(rx, ry, rw, rh);

    if (!UpscaleQuad) {
      static const char* kUpscaleFs = R"glsl(//VTK::System::Dec
in vec2 texCoord;
out vec4 fragColor;
uniform sampler2D lowResSrc;
void main() { fragColor = texture(lowResSrc, texCoord); }
)glsl";
      UpscaleQuad = std::make_unique<vtkOpenGLQuadHelper>(renWin, nullptr,
                                                          kUpscaleFs, nullptr);
    }
    renWin->GetShaderCache()->ReadyShaderProgram(UpscaleQuad->Program);
    const int upsUnit = 30;  // clear of VTK's allocated units this frame
    glActiveTexture(GL_TEXTURE0 + upsUnit);
    glBindTexture(GL_TEXTURE_2D, LowResTex);
    UpscaleQuad->Program->SetUniformi("lowResSrc", upsUnit);
    state->vtkglEnable(GL_BLEND);
    state->vtkglBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
                                  GL_ONE_MINUS_SRC_ALPHA);
    UpscaleQuad->Render();
    glActiveTexture(GL_TEXTURE0 + upsUnit);
    glBindTexture(GL_TEXTURE_2D, 0);
  } else {
    state->vtkglEnable(GL_BLEND);
    state->vtkglBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
                                  GL_ONE_MINUS_SRC_ALPHA);
    Quad->Render();
  }

  // Surface pass: opaque mesh, depth-tested, drawn over the composited
  // volume (the raymarch writes no depth). Shares the bound textures and
  // the streaming request SSBO.
  RenderSurface(renWin, ren);
  if (clusterView) RenderClusters(renWin, ren);

  if (timeThisFrame) {
    glEndQuery(GL_TIME_ELAPSED);
    GpuTimerInFlight = true;
  }

  state->vtkglDepthMask(GL_TRUE);
  state->vtkglEnable(GL_DEPTH_TEST);

  // Fence this frame's writes, then harvest the other buffer if its fence
  // has retired (zero-stall: skip rather than wait).
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
  if (FeedbackFence[FeedbackWriteIndex])
    glDeleteSync(static_cast<GLsync>(FeedbackFence[FeedbackWriteIndex]));
  FeedbackFence[FeedbackWriteIndex] =
      glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
  const int readIndex = 1 - FeedbackWriteIndex;
  ReadFeedback(readIndex);
  FeedbackWriteIndex = readIndex;

  Cache->poolTexture()->Deactivate();
  for (int i = 0; i < Cache->levelCount(); ++i)
    Cache->directoryTexture(i)->Deactivate();
  Cache->tilePoolTexture()->Deactivate();
  Cache->occPoolTexture()->Deactivate();
  if (Overlay && Overlay->initialized()) {
    Overlay->poolTexture()->Deactivate();
    for (int i = 0; i < Overlay->levelCount(); ++i)
      Overlay->directoryTexture(i)->Deactivate();
    Overlay->tilePoolTexture()->Deactivate();
  }
}

}  // namespace sv::render
