#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace sv::data {

struct SnicCluster {
  float z = 0, y = 0, x = 0;  // centroid, voxel coords of the input array
  float mean = 0;             // mean intensity, normalized 0..1
  std::uint32_t count = 0;    // voxels in the cluster
};

// 3D SNIC supervoxels (Achanta & Süsstrunk, "Superpixels and Polygons using
// Simple Non-Iterative Clustering", CVPR 2017, lifted from 4-connected
// pixels to 6-connected voxels). One priority-flood pass from grid seeds:
// the globally closest (voxel, cluster) pair in joint spatial+intensity
// space is committed first, the cluster centroid/mean update online, and
// the voxel's unlabeled neighbors are enqueued against the updated cluster.
// Non-iterative, exact connectivity by construction.
//
// seedFraction: clusters as a fraction of voxel count (the grid step is
// cbrt(1/seedFraction)). compactness: intensity-vs-space weight; smaller
// values let clusters hug intensity boundaries more tightly.
//
// Voxels at or below minIntensity (air: downsampled levels of "masked"
// volumes bottom out near — but not at — zero) are excluded entirely: they
// seed nothing, join nothing, and clusters never flood through them.
std::vector<SnicCluster> snic3d(std::span<const std::uint8_t> vol,
                                const std::array<std::uint32_t, 3>& dimsZyx,
                                double seedFraction, float compactness,
                                std::uint8_t minIntensity = 0);

// Otsu's threshold over the intensity histogram: the split maximizing
// between-class variance. Separates background from material without
// assuming where "air" sits — scans normalize it anywhere from 0 to ~103.
std::uint8_t otsuThreshold(std::span<const std::uint8_t> vol);

// A cluster lifted to world space for rendering: sphere center (world x,y,z
// where world units are level-0 voxels), radius of the equivalent-volume
// sphere, and the mean intensity as color value.
struct WorldSphere {
  float x, y, z, radius, value;
};

// voxelScale: world units per voxel of the segmented level (2^level for
// Vesuvius pyramids). Clusters smaller than minCount are dropped.
std::vector<WorldSphere> clustersToWorldSpheres(
    std::span<const SnicCluster> clusters, float voxelScale,
    std::uint32_t minCount);

}  // namespace sv::data
