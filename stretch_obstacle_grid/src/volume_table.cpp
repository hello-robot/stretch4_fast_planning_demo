#include "stretch_obstacle_grid/volume_table.hpp"

#include <algorithm>
#include <cmath>

#include <omp.h>

namespace stretch_obstacle_grid
{

namespace
{
constexpr float kTwoPi = 2.0f * static_cast<float>(M_PI);
}

VolumeTable::VolumeTable(const CylindricalGrid & grid)
: grid_(grid)
{
  const int zn = grid.z_bins();
  for (const Band & band : grid.bands()) {
    const size_t rn = band.r_end - band.r_begin;
    sums_.emplace_back((rn + 1) * (band.theta_bins + 1) * (zn + 1), 0u);
  }
}

size_t VolumeTable::sum_index(int band, int r, int t, int z) const
{
  const size_t tn = grid_.bands()[band].theta_bins + 1;
  const size_t zn = grid_.z_bins() + 1;
  return (static_cast<size_t>(r) * tn + t) * zn + z;
}

void VolumeTable::build(const std::vector<uint8_t> & values, uint32_t mask, int threads)
{
  if (threads <= 0) {
    threads = omp_get_max_threads();
  }
  const int zn = grid_.z_bins();
  const auto & bands = grid_.bands();
  for (size_t b = 0; b < bands.size(); ++b) {
    const Band & band = bands[b];
    const int rn = band.r_end - band.r_begin;
    const int tn = band.theta_bins;
    std::vector<uint32_t> & s = sums_[b];

    // Prefix sums along z, then theta, then r. Each pass adds whole z columns (contiguous
    // in memory) to the next, so the inner loops run straight through memory and vectorize.
    const size_t column = zn + 1;               // one padded z column
    const size_t plane = (tn + 1) * column;     // one padded radial bin

    // z, within each (r, theta) column; the radial bins are independent.
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int r = 0; r < rn; ++r) {
      for (int t = 0; t < tn; ++t) {
        const uint8_t * in = &values[band.offset + (static_cast<size_t>(r) * tn + t) * zn];
        uint32_t * out = &s[sum_index(b, r + 1, t + 1, 1)];
        uint32_t running = 0;
        for (int z = 0; z < zn; ++z) {
          running += (mask >> in[z]) & 1u;
          out[z] = running;
        }
      }
    }
    // theta, within each radial bin; the radial bins are independent.
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int r = 1; r <= rn; ++r) {
      uint32_t * rows = &s[r * plane];
      for (int t = 1; t <= tn; ++t) {
        uint32_t * row = rows + t * column;
        const uint32_t * previous = row - column;
        for (int z = 1; z <= zn; ++z) {
          row[z] += previous[z];
        }
      }
    }
    // r, one radial bin after another; split the theta columns between threads.
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int t = 1; t <= tn; ++t) {
      for (int r = 1; r <= rn; ++r) {
        uint32_t * row = &s[r * plane + t * column];
        const uint32_t * previous = row - plane;
        for (int z = 1; z <= zn; ++z) {
          row[z] += previous[z];
        }
      }
    }
  }
}

uint32_t VolumeTable::count(int band, int r0, int r1, int t0, int t1, int z0, int z1) const
{
  if (r0 >= r1 || t0 >= t1 || z0 >= z1) {
    return 0;
  }
  const int begin = grid_.bands()[band].r_begin;
  r0 -= begin;
  r1 -= begin;
  const std::vector<uint32_t> & s = sums_[band];
  auto at = [&](int r, int t, int z) {return static_cast<int64_t>(s[sum_index(band, r, t, z)]);};
  const int64_t total =
    at(r1, t1, z1) - at(r0, t1, z1) - at(r1, t0, z1) - at(r1, t1, z0) +
    at(r0, t0, z1) + at(r0, t1, z0) + at(r1, t0, z0) - at(r0, t0, z0);
  return static_cast<uint32_t>(total);
}

template<typename Fn>
void VolumeTable::for_each_range(const CylinderBox & box, Fn && fn) const
{
  const GridConfig & config = grid_.config();
  const float cell = config.cell;

  // Radial and vertical bins touched, clipped to the grid.
  const int r_lo = std::max(0, static_cast<int>(std::floor((box.r_min - config.r_min) / cell)));
  const int r_hi = std::min(grid_.radial_bins(), static_cast<int>(std::floor((box.r_max - config.r_min) / cell)) + 1);
  const int z_lo = std::max(0, static_cast<int>(std::floor((box.z_min - config.z_min) / cell)));
  const int z_hi = std::min(grid_.z_bins(), static_cast<int>(std::floor((box.z_max - config.z_min) / cell)) + 1);
  if (r_lo >= r_hi || z_lo >= z_hi || box.theta_max < box.theta_min) {
    return;
  }

  float start = std::fmod(box.theta_min, kTwoPi);
  if (start < 0.0f) {start += kTwoPi;}
  const float span = std::min(box.theta_max - box.theta_min, kTwoPi);

  const auto & bands = grid_.bands();
  for (size_t b = 0; b < bands.size(); ++b) {
    const Band & band = bands[b];
    const int r0 = std::max(r_lo, band.r_begin);
    const int r1 = std::min(r_hi, band.r_end);
    if (r0 >= r1) {
      continue;
    }
    const int n = band.theta_bins;
    const int first = std::min(n - 1, static_cast<int>(start / band.dtheta));
    const int last = static_cast<int>((start + span) / band.dtheta);  // may pass n: wraps
    if (last - first + 1 >= n) {
      fn(static_cast<int>(b), r0, r1, 0, n, z_lo, z_hi);
    } else if (last < n) {
      fn(static_cast<int>(b), r0, r1, first, last + 1, z_lo, z_hi);
    } else {
      fn(static_cast<int>(b), r0, r1, first, n, z_lo, z_hi);
      fn(static_cast<int>(b), r0, r1, 0, last - n + 1, z_lo, z_hi);
    }
  }
}

uint32_t VolumeTable::count(const CylinderBox & box) const
{
  uint32_t total = 0;
  for_each_range(box, [&](int b, int r0, int r1, int t0, int t1, int z0, int z1) {
      total += count(b, r0, r1, t0, t1, z0, z1);
    });
  return total;
}

void VolumeTable::voxels(const CylinderBox & box, std::vector<size_t> * out) const
{
  for_each_range(box, [&](int, int r0, int r1, int t0, int t1, int z0, int z1) {
      for (int r = r0; r < r1; ++r) {
        for (int t = t0; t < t1; ++t) {
          for (int z = z0; z < z1; ++z) {
            out->push_back(grid_.index(r, t, z));
          }
        }
      }
    });
}

}  // namespace stretch_obstacle_grid
