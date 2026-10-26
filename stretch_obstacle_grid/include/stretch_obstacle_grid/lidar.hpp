// What one LiDAR says about each voxel.
//
// Each frame becomes a range image: rows are the LiDAR's rings in elevation order, columns
// are azimuth bins, and each cell holds the nearest return in it and what that return hit.
// A RayTable, built once per LiDAR, lists the cells whose rays cross each voxel and how far
// the voxel is from the LiDAR. Comparing the two says whether the rays went past the voxel
// (free), ended in it (hit), or ended before it (hidden, and by what).

#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Geometry>

#include "stretch_obstacle_grid/cylindrical_grid.hpp"

namespace stretch_obstacle_grid
{

struct LidarModel
{
  Eigen::Isometry3f base_from_lidar;   // the LiDAR's frame in base_footprint
  std::vector<float> ring_elevation;   // rad in the LiDAR's frame, indexed by the ring field
  int azimuth_bins = 720;              // range image columns
};

// What the nearest return in a range image cell hit. kPassed: the return was dropped (a veil
// point), but the beam got at least as far as `range` before it, so the space in front is free
// and the space behind unknown.
enum class Hit : uint8_t {kNone = 0, kOther, kSelf, kFloor, kPassed};

struct RangeImage
{
  int rows = 0, cols = 0;
  std::vector<float> range;  // m; NaN where no return landed in the cell (and no kPassed)
  std::vector<Hit> hit;

  void reset(int rows, int cols);

  // Fill isolated dropouts. About 15% of rays randomly return nothing in any one frame, even
  // off ordinary surfaces. An empty cell with at least `min_neighbors` of its 8 neighbors
  // filled takes the nearest of them (range and hit), so voxels count as free only up to
  // the closest neighboring return. Larger empty patches, like a dark object that returns
  // nothing at all, stay empty.
  void fill_dropouts(int min_neighbors);
};

// One LiDAR's view of a voxel, combined over the rays that cross it.
enum class RayView : uint8_t
{
  kNoInfo = 0,     // outside the field of view, or at least half the crossing rays had no return
  kFree,           // more crossing rays went past it than had no return, and none ended in it
  kHit,            // a ray ended in it, on something other than the robot or floor
  kHitSelf,        // a ray ended in it, on the robot
  kHiddenBySelf,   // a ray ended before it, on the robot
  kHiddenByOther,  // a ray ended before it, on something else
};

class RayTable
{
public:
  RayTable(const CylindricalGrid & grid, const LidarModel & lidar);

  int rows() const {return static_cast<int>(row_elevation_.size());}
  int cols() const {return cols_;}
  // Range image row for a ring index, and column for an azimuth (rad, LiDAR frame).
  int row_of_ring(int ring) const {return row_of_ring_[ring];}
  int col_of_azimuth(float azimuth) const;
  // Unit direction (LiDAR frame) through the middle of a range image cell.
  Eigen::Vector3f cell_direction(int row, int col) const;

  // `margin` absorbs range noise: a return within margin of the voxel counts as ending in it.
  RayView view(size_t voxel, const RangeImage & image, float margin) const;

private:
  struct Entry
  {
    uint8_t row_lo, row_hi;  // row_lo == kInvisible: no ray crosses the voxel
    uint16_t col_lo, col_hi; // col_hi < col_lo: the span wraps past the last column
    float near, far;         // m from the LiDAR
  };
  static constexpr uint8_t kInvisible = 255;

  Entry make_entry(const CylindricalGrid & grid, const Eigen::Isometry3f & lidar_from_base, size_t voxel) const;

  int cols_;
  std::vector<int> row_of_ring_;
  std::vector<float> row_elevation_;  // ascending
  std::vector<Entry> entries_;        // per voxel
};

}  // namespace stretch_obstacle_grid
