// Cylindrical voxel grid around the base's rotation axis.
//
// The grid lives in base_footprint: the axis is the frame's z axis, z = 0 is the floor and
// theta = atan2(y, x). Voxels are `cell` deep in r and z. In theta, each radial band keeps a
// power-of-two number of bins, doubling outward so the arc length stays between one and two
// cells. That keeps every voxel roughly cubical without an octree.
//
// Within a band, voxels are stored as [radial][theta][z] with z fastest, so a voxel index
// is offset + ((ri - r_begin) * theta_bins + ti) * z_bins + zi.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace stretch_obstacle_grid
{

struct GridConfig
{
  float r_min = 0.02f;  // m, inner radius; closer to the axis the theta bins get too small
  // m, outer radius: the robot's reach (1.05 m with the arm out and the gripper straight)
  // plus how far a person can move while it reacts (about 1.5 m/s x 0.25 s worst-case
  // latency, plus braking)
  float r_max = 1.60f;
  float z_min = 0.04f;  // m above the floor; lower returns are floor
  float z_max = 2.00f;  // m
  float cell = 0.02f;   // m, radial and vertical voxel size, and the shortest arc length
};

// A ring of radial bins that share one theta resolution.
struct Band
{
  int theta_bins;      // power of two
  float dtheta;        // rad per theta bin
  int r_begin, r_end;  // global radial bin range [r_begin, r_end)
  size_t offset;       // index of the band's first voxel
};

// A voxel's position as (global radial bin, theta bin within its band, z bin).
struct Cell
{
  int ri, ti, zi;
};

class CylindricalGrid
{
public:
  explicit CylindricalGrid(const GridConfig & config = GridConfig());

  const GridConfig & config() const {return config_;}
  int radial_bins() const {return radial_bins_;}
  int z_bins() const {return z_bins_;}
  size_t size() const {return size_;}
  const std::vector<Band> & bands() const {return bands_;}
  const Band & band_of_radial(int ri) const {return bands_[band_index_[ri]];}
  int band_index_of_radial(int ri) const {return band_index_[ri];}

  size_t index(int ri, int ti, int zi) const;
  Cell cell(size_t index) const;

  // Bin of a coordinate, or -1 if it's outside the grid.
  int radial_bin(float r) const;
  int z_bin(float z) const;
  static int theta_bin(const Band & band, float theta);

  // Index of the voxel containing (x, y, z) in base_footprint; false if it's outside.
  bool locate(float x, float y, float z, size_t * index) const;

  // Voxel extents: radius [r0, r1], theta [t0, t1] (t0 in [0, 2pi)), z [z0, z1].
  void bounds(const Cell & c, float * r0, float * r1, float * t0, float * t1, float * z0, float * z1) const;
  void center(size_t index, float * x, float * y, float * z) const;

  // The voxels next to `index` across each face, including across band boundaries,
  // where a voxel faces two outer neighbors or shares one inner neighbor. Appends to `out`.
  void neighbors(size_t index, std::vector<size_t> * out) const;

private:
  GridConfig config_;
  int radial_bins_;
  int z_bins_;
  size_t size_;
  std::vector<Band> bands_;
  std::vector<int> band_index_;  // per radial bin
};

}  // namespace stretch_obstacle_grid
