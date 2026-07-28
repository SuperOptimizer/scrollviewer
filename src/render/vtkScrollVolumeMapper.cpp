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

std::string vtkScrollVolumeMapper::BuildFragmentShader() const {
  const int levels = Cache ? Cache->levelCount() : 1;
  // Per-level page-table samplers are distinct uniforms (GLSL cannot index a
  // sampler array with a non-uniform value); a switch helper dispatches.
  std::string dirDecls, dirCases;
  for (int i = 0; i < levels; ++i) {
    dirDecls += std::format("uniform usampler3D pageDir{};\n", i);
    dirCases += std::format(
        "    case {}: return texelFetch(pageDir{}, c, 0).r;\n", i, i);
  }

  return std::format(R"glsl(//VTK::System::Dec
#extension GL_ARB_shader_storage_buffer_object : require
in vec2 texCoord;
out vec4 fragColor;

// Ray-guided streaming: bricks wanted at a finer level than was resident.
layout(std430, binding = 0) buffer RequestBuf {{
  uint reqCount;
  uint reqPad0, reqPad1, reqPad2;
  uvec2 reqs[];  // (level, z<<20|y<<10|x)
}};

uniform sampler3D brickPool;
{0}
uniform mat4 invViewProj;      // NDC -> world
uniform vec3 volSizeWorld;     // full-res extent * spacing (x,y,z)
uniform vec3 levelShape[{1}];  // voxels per level (x,y,z)
uniform vec3 gridDim[{1}];     // brick-grid dims per level (x,y,z)
uniform vec3 poolTexels;       // pool texture size in texels
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

const int kLevels = {1};

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
  if (d == 0u) return 0u;
  if (d == 1u) return 0x80000000u;
  return texelFetch(tilePool, tileTexel(d - 2u, brick), 0).r;
}}

vec2 fetchOcc(int level, ivec3 brick) {{
  uint d = fetchDir(level, brick >> 4);
  if (d < 2u) return vec2(0.0, 1.0);
  return texelFetch(occPool, tileTexel(d - 2u, brick), 0).rg;
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
  float t = t0 + jitter(gl_FragCoord.xy) * dtMin;
  vec4 acc = vec4(0.0);
  bool requested = false;  // at most one streaming request per fragment

  // Brick-marching: resolve the page table once per brick, then inner-loop
  // through the brick's samples straight from the pool texture.
  for (int outer = 0; outer < 1024 && t < t1 && acc.a < 0.98; ++outer) {{
    // Cone LOD at the brick entry point (GigaVoxels-style), never finer
    // than the frame's minimum.
    float wpp = wppConst + wppPerDist * t;
    int lod = clamp(int(floor(log2(max(1.0, wpp / voxelWorld0)))), minLod,
                    kLevels - 1);
    vec3 p = (ro + t * rd) / volSizeWorld;

    // Fallback walk. EMPTY entries chain upward: the coarsest contiguous
    // empty level wins so one jump can cross a huge region (a deep-zoom ray
    // otherwise crawls across hundreds of fine empty bricks). RESIDENT stops
    // the walk (finer empties still take precedence over coarser content).
    int L = -1;
    uint e = 0u;
    ivec3 brick = ivec3(0);
    for (int i = lod; i < kLevels; ++i) {{
      vec3 v = p * levelShape[i];
      ivec3 b = ivec3(clamp(floor(v / chunkDim), vec3(0.0), gridDim[i] - 1.0));
      uint pe = fetchPageEntry(i, b);
      if ((pe & 0x80000000u) != 0u) {{
        L = i;
        e = pe;
        brick = b;
        continue;  // empty: try to widen the jump at a coarser level
      }}
      if ((pe & 0x40000000u) != 0u) {{
        if (L < 0) {{
          L = i;
          e = pe;
          brick = b;
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
    if (!emptyBrick) {{
      vec2 mm = fetchOcc(L, brick);
      emptyBrick = mm.y <= occSkipThreshold;
    }}
    if (emptyBrick) {{
      t = max(tExit, t) + dt * 0.05;
      continue;
    }}

    if (L > lod && !requested) {{
      // Sampling coarser than wanted: ask for the finer brick.
      vec3 vw = p * levelShape[lod];
      ivec3 want = ivec3(clamp(floor(vw / chunkDim), vec3(0.0),
                               gridDim[lod] - 1.0));
      uint idx = atomicAdd(reqCount, 1u);
      if (idx < {3}u)
        reqs[idx] = uvec2(uint(lod), (uint(want.z) << 20) |
                                     (uint(want.y) << 10) | uint(want.x));
      requested = true;
    }}

    // March inside this resident brick: no more page-table traffic.
    vec3 poolBase = vec3(float(e & 0x3FFu), float((e >> 10) & 0x3FFu),
                         float((e >> 20) & 0x3FFu)) * borderedDim + 1.0;
    float stepRel = dt / dtMin;
    float tEnd = min(tExit, t1);
    int sRun = 0;
    for (; sRun < 160 && t < tEnd && acc.a < 0.98; ++sRun) {{
      vec3 v = ((ro + t * rd) / volSizeWorld) * levelShape[L];
      vec3 inBrick = clamp(v - vec3(brick) * chunkDim, vec3(0.0),
                           vec3(chunkDim));
      float dens = texture(brickPool, (poolBase + inBrick) / poolTexels).r;
      float w = clamp((dens - winCenter) / winWidth + 0.5, 0.0, 1.0);
      float a = 1.0 - pow(1.0 - w * opacityScale, stepRel);
      acc.rgb += (1.0 - acc.a) * a * vec3(w);
      acc.a   += (1.0 - acc.a) * a;
      t += dt;
    }}
    // The final t already sits < dt into the next brick, so sampling stays
    // continuous across the boundary. Nudge only on a zero-sample stall
    // (float-precision entry exactly on a face).
    if (sRun == 0) t = max(t, tExit) + dt * 0.05;
  }}
  if (acc.a < 0.004) discard;
  fragColor = acc;
}}
)glsl",
                     dirDecls, levels, dirCases, kMaxFeedback);
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

  if (!Cache->initialized()) Cache->initialize(renWin);
  Cache->frameBegin(++FrameIndex);

  if (PreRender) PreRender();  // drain pipeline, upload bricks (context is current)
  Cache->syncPageTables();

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
  prog->SetUniformf("opacityScale", OpacityScale);
  // Bricks whose max density maps to zero opacity under the current window
  // are culled exactly like empty ones.
  prog->SetUniformf("occSkipThreshold",
                    std::max(0.f, Level - 0.5f * Window));

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

  const bool lowRes = RenderScale < 0.999f;
  if (lowRes) {
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
}

}  // namespace sv::render
