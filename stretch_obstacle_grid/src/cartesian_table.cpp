#include "stretch_obstacle_grid/cartesian_table.hpp"

#include <algorithm>
#include <cmath>

namespace stretch_obstacle_grid
{

CartesianTable::CartesianTable(const CylindricalGrid & grid, float cell)
: cell_(cell)
{
  const GridConfig & config = grid.config();
  origin_ = Eigen::Vector3f(-config.r_max, -config.r_max, config.z_min);
  nx_ = ny_ = static_cast<int>(std::ceil(2.0f * config.r_max / cell));
  nz_ = static_cast<int>(std::ceil((config.z_max - config.z_min) / cell));

  // A voxel's farthest point from its center: half its depth and height, and half its outer
  // arc (the chord is shorter, so this is a safe bound).
  float half_arc = 0.0f;
  for (const Band & band : grid.bands()) {
    half_arc = std::max(half_arc, 0.5f * (config.r_min + band.r_end * config.cell) * band.dtheta);
  }
  const float half = 0.5f * config.cell;
  reach_ = std::sqrt(2.0f * half * half + half_arc * half_arc);

  cell_of_voxel_.resize(grid.size());
  for (size_t i = 0; i < grid.size(); ++i) {
    Eigen::Vector3f c;
    grid.center(i, &c.x(), &c.y(), &c.z());
    const Eigen::Vector3f at = (c - origin_) / cell;
    const int x = std::clamp(static_cast<int>(at.x()), 0, nx_ - 1);
    const int y = std::clamp(static_cast<int>(at.y()), 0, ny_ - 1);
    const int z = std::clamp(static_cast<int>(at.z()), 0, nz_ - 1);
    cell_of_voxel_[i] = (x * ny_ + y) * nz_ + z;
  }
  sums_.assign(static_cast<size_t>(nx_ + 1) * (ny_ + 1) * (nz_ + 1), 0);
}

void CartesianTable::build(const std::vector<uint8_t> & values, uint32_t mask)
{
  std::fill(sums_.begin(), sums_.end(), 0u);
  for (size_t i = 0; i < values.size(); ++i) {
    if (mask & (1u << values[i])) {
      const int32_t c = cell_of_voxel_[i];
      const int z = c % nz_, y = (c / nz_) % ny_, x = c / (nz_ * ny_);
      ++sums_[sum_index(x + 1, y + 1, z + 1)];
    }
  }
  // Prefix sums along z, then y, then x.
  for (int x = 1; x <= nx_; ++x) {
    for (int y = 1; y <= ny_; ++y) {
      uint32_t * column = &sums_[sum_index(x, y, 0)];
      for (int z = 1; z <= nz_; ++z) {column[z] += column[z - 1];}
    }
  }
  for (int x = 1; x <= nx_; ++x) {
    for (int y = 1; y <= ny_; ++y) {
      uint32_t * column = &sums_[sum_index(x, y, 0)];
      const uint32_t * previous = &sums_[sum_index(x, y - 1, 0)];
      for (int z = 1; z <= nz_; ++z) {column[z] += previous[z];}
    }
  }
  for (int x = 1; x <= nx_; ++x) {
    for (int y = 1; y <= ny_; ++y) {
      uint32_t * column = &sums_[sum_index(x, y, 0)];
      const uint32_t * previous = &sums_[sum_index(x - 1, y, 0)];
      for (int z = 1; z <= nz_; ++z) {column[z] += previous[z];}
    }
  }
}

uint32_t CartesianTable::count(const Eigen::AlignedBox3f & box) const
{
  // Cells holding the center of any voxel that overlaps the box.
  const Eigen::Vector3f lo = (box.min().array() - reach_ - origin_.array()) / cell_;
  const Eigen::Vector3f hi = (box.max().array() + reach_ - origin_.array()) / cell_;
  const int x0 = std::max(0, static_cast<int>(std::floor(lo.x())));
  const int y0 = std::max(0, static_cast<int>(std::floor(lo.y())));
  const int z0 = std::max(0, static_cast<int>(std::floor(lo.z())));
  const int x1 = std::min(nx_, static_cast<int>(std::floor(hi.x())) + 1);
  const int y1 = std::min(ny_, static_cast<int>(std::floor(hi.y())) + 1);
  const int z1 = std::min(nz_, static_cast<int>(std::floor(hi.z())) + 1);
  if (x0 >= x1 || y0 >= y1 || z0 >= z1) {
    return 0;
  }
  auto at = [&](int x, int y, int z) {return static_cast<int64_t>(sums_[sum_index(x, y, z)]);};
  return static_cast<uint32_t>(
    at(x1, y1, z1) - at(x0, y1, z1) - at(x1, y0, z1) - at(x1, y1, z0) +
    at(x0, y0, z1) + at(x0, y1, z0) + at(x1, y0, z0) - at(x0, y0, z0));
}

}  // namespace stretch_obstacle_grid
