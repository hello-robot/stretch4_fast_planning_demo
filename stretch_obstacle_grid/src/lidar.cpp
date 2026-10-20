#include "stretch_obstacle_grid/lidar.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace stretch_obstacle_grid
{

namespace
{
constexpr float kPi = static_cast<float>(M_PI);
constexpr float kTwoPi = 2.0f * kPi;
// A voxel between two rings, with no ring inside it, still gets the nearest ring if that
// ring is at most this far (rad) outside the voxel; past the outermost rings it's unseen.
constexpr float kRingReach = 0.035f;  // ~2 deg, a bit more than the widest ring gap
}  // namespace

void RangeImage::reset(int new_rows, int new_cols)
{
  rows = new_rows;
  cols = new_cols;
  range.assign(static_cast<size_t>(rows) * cols, std::numeric_limits<float>::quiet_NaN());
  hit.assign(static_cast<size_t>(rows) * cols, Hit::kNone);
}

void RangeImage::fill_dropouts(int min_neighbors)
{
  const std::vector<float> original = range;
  const std::vector<Hit> original_hit = hit;
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const size_t cell = static_cast<size_t>(row) * cols + col;
      if (!std::isnan(original[cell])) {
        continue;
      }
      int filled = 0;
      float nearest = std::numeric_limits<float>::infinity();
      Hit nearest_hit = Hit::kNone;
      for (int dr = -1; dr <= 1; ++dr) {
        const int r = row + dr;
        if (r < 0 || r >= rows) {
          continue;
        }
        for (int dc = -1; dc <= 1; ++dc) {
          const size_t other = static_cast<size_t>(r) * cols + (col + dc + cols) % cols;  // azimuth wraps
          if (other == cell || std::isnan(original[other])) {
            continue;
          }
          ++filled;
          if (original[other] < nearest) {
            nearest = original[other];
            nearest_hit = original_hit[other];
          }
        }
      }
      if (filled >= min_neighbors) {
        range[cell] = nearest;
        hit[cell] = nearest_hit;
      }
    }
  }
}

RayTable::RayTable(const CylindricalGrid & grid, const LidarModel & lidar)
: cols_(lidar.azimuth_bins)
{
  const int rings = static_cast<int>(lidar.ring_elevation.size());
  if (rings == 0 || rings >= kInvisible || cols_ <= 0 || cols_ > 65535) {
    throw std::invalid_argument("RayTable: bad ring or azimuth count");
  }
  std::vector<int> order(rings);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
      return lidar.ring_elevation[a] < lidar.ring_elevation[b];
    });
  row_of_ring_.resize(rings);
  for (int row = 0; row < rings; ++row) {
    row_of_ring_[order[row]] = row;
    row_elevation_.push_back(lidar.ring_elevation[order[row]]);
  }

  const Eigen::Isometry3f lidar_from_base = lidar.base_from_lidar.inverse();
  entries_.resize(grid.size());
  #pragma omp parallel for schedule(static)
  for (size_t voxel = 0; voxel < grid.size(); ++voxel) {
    entries_[voxel] = make_entry(grid, lidar_from_base, voxel);
  }
}

int RayTable::col_of_azimuth(float azimuth) const
{
  const int col = static_cast<int>((azimuth + kPi) / kTwoPi * cols_);
  return std::clamp(col, 0, cols_ - 1);
}

Eigen::Vector3f RayTable::cell_direction(int row, int col) const
{
  const float elevation = row_elevation_[row];
  const float azimuth = -kPi + (col + 0.5f) * kTwoPi / cols_;
  return Eigen::Vector3f(
    std::cos(elevation) * std::cos(azimuth), std::cos(elevation) * std::sin(azimuth), std::sin(elevation));
}

RayTable::Entry RayTable::make_entry(
  const CylindricalGrid & grid, const Eigen::Isometry3f & lidar_from_base, size_t voxel) const
{
  float r0, r1, t0, t1, z0, z1;
  grid.bounds(grid.cell(voxel), &r0, &r1, &t0, &t1, &z0, &z1);

  // The voxel's corners and center, in the LiDAR's spherical coordinates. Azimuths are
  // taken relative to the center's so a voxel straddling +/-pi stays contiguous.
  const float rc = 0.5f * (r0 + r1), tc = 0.5f * (t0 + t1), zc = 0.5f * (z0 + z1);
  const Eigen::Vector3f center = lidar_from_base * Eigen::Vector3f(rc * std::cos(tc), rc * std::sin(tc), zc);
  const float center_range = center.norm();
  const float center_azimuth = std::atan2(center.y(), center.x());
  const float center_elevation = std::asin(center.z() / center_range);

  Entry entry{kInvisible, kInvisible, 0, 0, center_range, center_range};
  float el_lo = center_elevation, el_hi = center_elevation;
  float az_lo = 0.0f, az_hi = 0.0f;  // relative to center_azimuth
  for (float r : {r0, r1}) {
    for (float t : {t0, t1}) {
      for (float z : {z0, z1}) {
        const Eigen::Vector3f p = lidar_from_base * Eigen::Vector3f(r * std::cos(t), r * std::sin(t), z);
        const float range = p.norm();
        entry.near = std::min(entry.near, range);
        entry.far = std::max(entry.far, range);
        const float elevation = std::asin(p.z() / range);
        el_lo = std::min(el_lo, elevation);
        el_hi = std::max(el_hi, elevation);
        const float azimuth = std::remainder(std::atan2(p.y(), p.x()) - center_azimuth, kTwoPi);
        az_lo = std::min(az_lo, azimuth);
        az_hi = std::max(az_hi, azimuth);
      }
    }
  }

  // Rows: the rings inside the voxel's elevation span, or else the nearest ring if it's close.
  const auto first = std::lower_bound(row_elevation_.begin(), row_elevation_.end(), el_lo);
  const auto last = std::upper_bound(row_elevation_.begin(), row_elevation_.end(), el_hi);
  if (first < last) {
    entry.row_lo = static_cast<uint8_t>(first - row_elevation_.begin());
    entry.row_hi = static_cast<uint8_t>(last - row_elevation_.begin() - 1);
  } else {
    // No ring inside: `first` is the first ring above the voxel.
    int nearest = -1;
    float gap = kRingReach;
    if (first != row_elevation_.end() && *first - el_hi <= gap) {
      nearest = static_cast<int>(first - row_elevation_.begin());
      gap = *first - el_hi;
    }
    if (first != row_elevation_.begin() && el_lo - *(first - 1) <= gap) {
      nearest = static_cast<int>(first - row_elevation_.begin()) - 1;
    }
    if (nearest < 0) {
      return entry;  // outside the field of view
    }
    entry.row_lo = entry.row_hi = static_cast<uint8_t>(nearest);
  }

  // Columns whose centers fall inside the azimuth span, or else the one containing the
  // voxel's center. A voxel around the LiDAR's pole spans every column.
  const float width = kTwoPi / cols_;
  if (az_hi - az_lo >= kPi) {
    entry.col_lo = 0;
    entry.col_hi = static_cast<uint16_t>(cols_ - 1);
    return entry;
  }
  const float start = center_azimuth + az_lo + kPi;  // in [0, 3pi) before wrapping
  const float end = center_azimuth + az_hi + kPi;
  int lo = static_cast<int>(std::ceil(start / width - 0.5f));
  int hi = static_cast<int>(std::floor(end / width - 0.5f));
  if (lo > hi) {
    lo = hi = col_of_azimuth(center_azimuth);
  }
  entry.col_lo = static_cast<uint16_t>(((lo % cols_) + cols_) % cols_);
  entry.col_hi = static_cast<uint16_t>(((hi % cols_) + cols_) % cols_);
  return entry;
}

RayView RayTable::view(size_t voxel, const RangeImage & image, float margin) const
{
  const Entry & e = entries_[voxel];
  if (e.row_lo == kInvisible || image.range.empty()) {
    return RayView::kNoInfo;
  }
  bool hit = false, hit_self = false, hidden_self = false;
  int no_info = 0, passed = 0;  // cells that say nothing about the voxel, and that went past it
  const int col_count = e.col_hi >= e.col_lo ? e.col_hi - e.col_lo + 1 : cols_ - e.col_lo + e.col_hi + 1;
  for (int row = e.row_lo; row <= e.row_hi; ++row) {
    const size_t row_start = static_cast<size_t>(row) * cols_;
    for (int k = 0, col = e.col_lo; k < col_count; ++k, col = (col + 1 == cols_ ? 0 : col + 1)) {
      const float range = image.range[row_start + col];
      if (std::isnan(range)) {
        ++no_info;
        continue;
      }
      // The floor bounds the grid from below, so ending on it never hides a voxel.
      const Hit what = image.hit[row_start + col];
      if (what == Hit::kFloor) {
        ++passed;
        continue;
      }
      if (what == Hit::kPassed) {
        // Free up to `range`; past it this beam says nothing.
        ++(range < e.far + margin ? no_info : passed);
        continue;
      }
      if (range < e.near - margin) {
        if (what == Hit::kOther) {
          return RayView::kHiddenByOther;  // the most conservative answer: stop here
        }
        hidden_self = true;
      } else if (range <= e.far + margin) {
        (what == Hit::kSelf ? hit_self : hit) = true;
      } else {
        ++passed;
      }
    }
  }
  if (hit) {return RayView::kHit;}
  if (hidden_self) {return RayView::kHiddenBySelf;}
  if (hit_self) {return RayView::kHitSelf;}
  // Most of the voxel's cells went past it: whatever the rest might hide is smaller than half
  // a voxel. The LiDAR returns nothing along many depth edges, in strips a few beams wide.
  return no_info >= passed ? RayView::kNoInfo : RayView::kFree;
}

}  // namespace stretch_obstacle_grid
