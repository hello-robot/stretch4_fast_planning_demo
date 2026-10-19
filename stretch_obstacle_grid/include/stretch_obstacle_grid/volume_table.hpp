// Summed-volume tables over a CylindricalGrid: count the flagged voxels in any
// (theta, r, z) box with eight lookups per band it touches.
//
// Since the arm extends radially through the rotation axis, the volume it sweeps while the
// base turns is a box in (theta, r, z), and so is a padded shape's footprint, so these
// counts answer "is anything in the way" without touching individual voxels.

#pragma once

#include <cstdint>
#include <vector>

#include "stretch_obstacle_grid/cylindrical_grid.hpp"

namespace stretch_obstacle_grid
{

// A box in cylindrical coordinates (base_footprint). theta runs from theta_min to
// theta_max counterclockwise; the span may cross 0 and is capped at a full turn.
struct CylinderBox
{
  float theta_min, theta_max;  // rad
  float r_min, r_max;          // m
  float z_min, z_max;          // m
};

class VolumeTable
{
public:
  explicit VolumeTable(const CylindricalGrid & grid);

  // Rebuild from one small value per voxel (a label): voxel i counts when bit values[i] of
  // `mask` is set. `threads` caps the OpenMP threads (0: the OpenMP default). The passes are
  // short, so past about 8 threads the synchronization costs more than it saves.
  void build(const std::vector<uint8_t> & values, uint32_t mask, int threads = 0);

  // Flagged voxels in bins [r0, r1) x [t0, t1) x [z0, z1) of one band (global radial bins,
  // band-local theta bins, no wrap).
  uint32_t count(int band, int r0, int r1, int t0, int t1, int z0, int z1) const;

  // Flagged voxels overlapping a metric box. Every voxel the box touches counts, so this
  // never misses anything inside the box.
  uint32_t count(const CylinderBox & box) const;

private:
  size_t sum_index(int band, int r, int t, int z) const;  // r band-local, all +1 padded

  const CylindricalGrid & grid_;
  std::vector<std::vector<uint32_t>> sums_;  // per band: (radial+1) x (theta+1) x (z+1)
};

}  // namespace stretch_obstacle_grid
