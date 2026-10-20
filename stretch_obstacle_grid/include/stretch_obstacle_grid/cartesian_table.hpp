// Summed-volume tables in x, y, z over a CylindricalGrid's voxels: count the flagged voxels
// overlapping any axis-aligned box (base_footprint).
//
// A VolumeTable answers boxes in (theta, r, z), which fit base rotation exactly. But near the
// axis a CylinderBox around a small box spans a wide range of theta, and grown by a few cm a
// full turn, so clearances there come out far too small. Boxes in x, y, z don't have that
// problem. Each voxel counts in the coarse cell holding its center, and a query is grown by the
// farthest a voxel reaches from its center, so every voxel overlapping the box is counted
// (along with some just outside it).

#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Geometry>

#include "stretch_obstacle_grid/cylindrical_grid.hpp"

namespace stretch_obstacle_grid
{

class CartesianTable
{
public:
  // `cell` (m): the table's cell size. Coarser is cheaper to build but rounds queries further out.
  explicit CartesianTable(const CylindricalGrid & grid, float cell = 0.04f);

  // Rebuild from one small value per voxel (a label): voxel i counts when bit values[i] of
  // `mask` is set.
  void build(const std::vector<uint8_t> & values, uint32_t mask);

  // Flagged voxels overlapping the box (and possibly some up to a cell and a voxel outside it).
  uint32_t count(const Eigen::AlignedBox3f & box) const;

  float cell() const {return cell_;}
  float voxel_reach() const {return reach_;}

private:
  size_t sum_index(int x, int y, int z) const {return (static_cast<size_t>(x) * (ny_ + 1) + y) * (nz_ + 1) + z;}

  float cell_;
  float reach_;             // farthest any part of a voxel is from its center
  Eigen::Vector3f origin_;  // corner of cell (0, 0, 0)
  int nx_, ny_, nz_;
  std::vector<int32_t> cell_of_voxel_;
  std::vector<uint32_t> sums_;  // (nx+1) x (ny+1) x (nz+1), +1 padded
};

}  // namespace stretch_obstacle_grid
