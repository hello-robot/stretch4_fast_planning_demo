#include "stretch_obstacle_grid/obstacle_grid.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>

namespace stretch_obstacle_grid
{

namespace
{
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start)
{
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
}  // namespace

ObstacleGrid::ObstacleGrid(
  const GridConfig & config, const std::vector<LidarModel> & lidars, const ObstacleGridParams & params)
: grid_(config), params_(params), lidars_(lidars), frames_(lidars.size()),
  static_self_flags_(grid_.size()), self_flags_(grid_.size()), occupied_by_(grid_.size()),
  labels_(grid_.size(), kUnknown),
  occupied_table_(grid_), unsafe_table_(grid_)
{
  if (lidar_count() > kMaxLidars) {
    throw std::invalid_argument("ObstacleGrid: at most 8 LiDARs");
  }
  for (const LidarModel & lidar : lidars_) {
    rays_.push_back(std::make_unique<RayTable>(grid_, lidar));
  }
}

void ObstacleGrid::set_static_self(const SelfModel & self)
{
  static_self_ = self;
  std::fill(static_self_flags_.begin(), static_self_flags_.end(), 0);
  static_self_.rasterize(grid_, params_.self_voxel_padding, &static_self_flags_);
}

bool ObstacleGrid::is_self(const SelfModel & moving, const Eigen::Vector3f & p) const
{
  return static_self_.contains(p, params_.self_point_padding) || moving.contains(p, params_.self_point_padding);
}

void ObstacleGrid::add_frame(
  int lidar, const float * xyz, const uint16_t * ring, size_t n, double stamp, const SelfModel & self)
{
  const auto start = Clock::now();
  const RayTable & rays = *rays_.at(lidar);
  const Eigen::Isometry3f & base_from_lidar = lidars_[lidar].base_from_lidar;
  const float floor_z = grid_.config().z_min;

  Frame & frame = frames_[lidar];
  frame.image.reset(rays.rows(), rays.cols());
  frame.occupied.clear();
  frame.stamp = stamp;

  auto point = [&](int32_t i) {return Eigen::Map<const Eigen::Vector3f>(&xyz[3 * i]);};
  const int rows = rays.rows(), cols = rays.cols();

  // Each range image cell's nearest return, skipping no-returns and dual-return duplicates.
  // Only that return counts from here on; the rest of a cell's returns are within half a
  // degree behind it.
  nearest_in_cell_.assign(static_cast<size_t>(rows) * cols, -1);
  const size_t stride = params_.duplicate_stride;
  for (size_t i = 0; i < n; ++i) {
    const float * p = &xyz[3 * i];
    if (stride > 0 && i >= stride && std::equal(p, p + 3, p - 3 * stride) && ring[i] == ring[i - stride]) {
      continue;  // the second return of a dual-return pair, same as the first
    }
    if (!point(i).allFinite() || point(i).squaredNorm() < params_.min_return_range * params_.min_return_range) {
      continue;  // no return, or too near to trust
    }
    int32_t & nearest = nearest_in_cell_[rays.row_of_ring(ring[i]) * cols + rays.col_of_azimuth(std::atan2(p[1], p[0]))];
    if (nearest < 0 || point(i).squaredNorm() < point(nearest).squaredNorm()) {
      nearest = static_cast<int32_t>(i);
    }
  }
  // Small clusters first: a stray return would otherwise mark its real neighbors behind it
  // as veil points.
  drop_small_clusters(xyz, base_from_lidar, rows, cols);
  drop_veil_points(xyz, rows, cols);

  for (size_t cell = 0; cell < nearest_in_cell_.size(); ++cell) {
    const int32_t i = nearest_in_cell_[cell];
    if (i < 0) {
      continue;
    }
    const Eigen::Vector3f in_base = base_from_lidar * point(i);
    float range = point(i).norm();
    Hit what = Hit::kOther;
    if (is_self(self, in_base)) {
      what = Hit::kSelf;
    } else {
      // A return from behind the robot along its own beam is a multipath ghost (the beam
      // bounced off the shiny gripper and came back late): the beam really ended on the robot.
      const Eigen::Vector3f direction = base_from_lidar.linear() * point(i) / range;
      const float robot = std::min(
        static_self_.ray_distance(base_from_lidar.translation(), direction),
        self.ray_distance(base_from_lidar.translation(), direction));
      if (robot < range - params_.range_margin) {
        what = Hit::kSelf;
        range = robot;
      } else if (in_base.z() < floor_z) {
        what = Hit::kFloor;
      }
    }
    frame.image.range[cell] = range;
    frame.image.hit[cell] = what;

    size_t voxel;
    if (what == Hit::kOther && grid_.locate(in_base.x(), in_base.y(), in_base.z(), &voxel)) {
      frame.occupied.push_back(voxel);
    }
  }
  if (params_.empty_rays_hit_self) {
    const Eigen::Vector3f origin = base_from_lidar.translation();
    for (int row = 0; row < rays.rows(); ++row) {
      for (int col = 0; col < rays.cols(); ++col) {
        const size_t cell = static_cast<size_t>(row) * rays.cols() + col;
        if (!std::isnan(frame.image.range[cell])) {
          continue;
        }
        const Eigen::Vector3f direction = base_from_lidar.linear() * rays.cell_direction(row, col);
        const float distance = std::min(
          static_self_.ray_distance(origin, direction), self.ray_distance(origin, direction));
        if (std::isfinite(distance)) {
          frame.image.range[cell] = distance;
          frame.image.hit[cell] = Hit::kSelf;
        }
      }
    }
  }
  frame.image.fill_dropouts(params_.dropout_min_neighbors);
  timing_.bin_points = ms_since(start);
}

void ObstacleGrid::drop_veil_points(const float * xyz, int rows, int cols)
{
  if (params_.veil_angle <= 0.0f) {
    return;
  }
  auto point = [&](int32_t i) {return Eigen::Map<const Eigen::Vector3f>(&xyz[3 * i]);};
  const float cos_angle = std::cos(params_.veil_angle);
  std::vector<uint8_t> & veil = veil_flags_;
  veil.assign(nearest_in_cell_.size(), 0);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const int32_t a = nearest_in_cell_[row * cols + col];
      if (a < 0) {
        continue;
      }
      const Eigen::Vector3f p = point(a);
      const Eigen::Vector3f toward_lidar = -p.normalized();
      for (int dr = -1; dr <= 1 && !veil[row * cols + col]; ++dr) {
        if (row + dr < 0 || row + dr >= rows) {
          continue;
        }
        for (int dc = -1; dc <= 1; ++dc) {
          const int32_t b = nearest_in_cell_[(row + dr) * cols + (col + dc + cols) % cols];
          if (b < 0 || b == a || point(b).squaredNorm() >= p.squaredNorm()) {
            continue;  // only nearer neighbors can hide this one
          }
          const Eigen::Vector3f to_neighbor = point(b) - p;
          if (toward_lidar.dot(to_neighbor) > cos_angle * to_neighbor.norm()) {
            veil[row * cols + col] = 1;
            break;
          }
        }
      }
    }
  }
  for (size_t c = 0; c < veil.size(); ++c) {
    if (veil[c]) {
      nearest_in_cell_[c] = -1;
    }
  }
}

void ObstacleGrid::drop_small_clusters(
  const float * xyz, const Eigen::Isometry3f & base_from_lidar, int rows, int cols)
{
  if (params_.min_cluster_size <= 1 && params_.min_cluster_rings <= 1) {
    return;
  }
  auto point = [&](int32_t i) {return Eigen::Map<const Eigen::Vector3f>(&xyz[3 * i]);};
  const size_t cells = nearest_in_cell_.size();

  // Floor returns stay out of the clusters: they never become obstacles, far away each
  // ring's floor returns are too spread out to link (dropping them would leave holes where
  // the floor proves the space above it is clear), and noise just above the floor mustn't
  // hide in the floor's cluster.
  const float floor_z = grid_.config().z_min;
  floor_cell_.assign(cells, 0);
  for (size_t c = 0; c < cells; ++c) {
    const int32_t i = nearest_in_cell_[c];
    floor_cell_[c] = i >= 0 && (base_from_lidar * point(i)).z() < floor_z;
  }

  cluster_parent_.resize(cells);
  for (size_t c = 0; c < cells; ++c) {
    cluster_parent_[c] = static_cast<uint32_t>(c);
  }
  auto root = [&](uint32_t c) {
    while (cluster_parent_[c] != c) {
      cluster_parent_[c] = cluster_parent_[cluster_parent_[c]];  // path halving
      c = cluster_parent_[c];
    }
    return c;
  };

  // Join each cell to its neighbors to the right and in the next row (azimuth wraps) when
  // their returns are close enough; that covers all 8 neighbors once.
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const uint32_t cell = row * cols + col;
      const int32_t a = nearest_in_cell_[cell];
      if (a < 0 || floor_cell_[cell]) {
        continue;
      }
      const float radius = params_.cluster_radius + params_.cluster_radius_per_m * point(a).norm();
      const std::array<std::pair<int, int>, 4> next = {{{0, 1}, {1, -1}, {1, 0}, {1, 1}}};
      for (const auto & [dr, dc] : next) {
        if (row + dr >= rows) {
          continue;
        }
        const uint32_t other = (row + dr) * cols + (col + dc + cols) % cols;
        const int32_t b = nearest_in_cell_[other];
        if (b >= 0 && !floor_cell_[other] && (point(a) - point(b)).squaredNorm() <= radius * radius) {
          cluster_parent_[root(cell)] = root(other);
        }
      }
    }
  }

  // Each cluster's size and the range of rings (rows) it spans.
  cluster_size_.assign(cells, 0);
  cluster_row_min_.assign(cells, rows);
  cluster_row_max_.assign(cells, 0);
  for (size_t c = 0; c < cells; ++c) {
    if (nearest_in_cell_[c] >= 0 && !floor_cell_[c]) {
      const uint32_t r = root(static_cast<uint32_t>(c));
      const uint16_t row = static_cast<uint16_t>(c / cols);
      ++cluster_size_[r];
      cluster_row_min_[r] = std::min(cluster_row_min_[r], row);
      cluster_row_max_[r] = std::max(cluster_row_max_[r], row);
    }
  }
  for (size_t c = 0; c < cells; ++c) {
    if (nearest_in_cell_[c] < 0 || floor_cell_[c]) {
      continue;
    }
    const uint32_t r = root(static_cast<uint32_t>(c));
    if (cluster_size_[r] < static_cast<uint32_t>(params_.min_cluster_size) ||
      cluster_row_max_[r] - cluster_row_min_[r] + 1 < params_.min_cluster_rings)
    {
      nearest_in_cell_[c] = -1;
    }
  }
}

double ObstacleGrid::oldest_stamp() const
{
  double oldest = std::numeric_limits<double>::infinity();
  for (const Frame & frame : frames_) {
    oldest = std::min(oldest, frame.stamp);
  }
  return oldest;
}

RayView ObstacleGrid::ray_view(int lidar, size_t voxel) const
{
  return rays_.at(lidar)->view(voxel, frames_[lidar].image, params_.range_margin);
}

Label ObstacleGrid::label_voxel(size_t voxel) const
{
  std::array<RayView, kMaxLidars> views;
  for (int lidar = 0; lidar < lidar_count(); ++lidar) {
    views[lidar] = ray_view(lidar, voxel);
  }

  // Occupied if some LiDAR has a return here that no newer frame from another LiDAR sees
  // straight through. The LiDARs aren't synchronized, so without this an obstacle that
  // has moved stays where the older frame saw it until that LiDAR's next frame.
  for (int lidar = 0; lidar < lidar_count(); ++lidar) {
    if (!(occupied_by_[voxel] >> lidar & 1u)) {
      continue;
    }
    bool stale = false;
    for (int other = 0; other < lidar_count() && !stale; ++other) {
      stale = frames_[other].stamp > frames_[lidar].stamp && views[other] == RayView::kFree;
    }
    if (!stale) {
      return kOccupied;
    }
  }
  if (self_flags_[voxel]) {return kSelf;}

  bool unknown = false, shadow = false;
  for (int lidar = 0; lidar < lidar_count(); ++lidar) {
    switch (views[lidar]) {
      case RayView::kFree:
        return kFree;  // one LiDAR seeing through it is enough
      case RayView::kHit:  // a return in it that wasn't binned here: the edge of an obstacle
      case RayView::kHiddenByOther:
        unknown = true;
        break;
      case RayView::kHitSelf:
      case RayView::kHiddenBySelf:
        shadow = true;
        break;
      case RayView::kNoInfo:
        break;
    }
  }
  // Hidden by something else from any LiDAR, or seen by none: unknown.
  return (shadow && !unknown) ? kShadow : kUnknown;
}

void ObstacleGrid::contaminate_shadows()
{
  // Breadth-first from every shadow voxel that touches unknown space or an obstacle, out to
  // contamination_reach through the connected shadow. Distance is counted in steps of one
  // cell; a step around theta can be up to two cells long, so contamination may reach a bit
  // farther around the axis than the parameter says.
  const float reach = params_.contamination_reach;
  const int max_steps = std::isfinite(reach) ?
    static_cast<int>(std::ceil(reach / grid_.config().cell)) : std::numeric_limits<int>::max();

  std::deque<std::pair<size_t, int>> queue;  // (voxel, steps from where the shadow was touched)
  std::vector<size_t> next_to;
  for (size_t voxel = 0; voxel < labels_.size(); ++voxel) {
    if (labels_[voxel] != kShadow) {
      continue;
    }
    next_to.clear();
    grid_.neighbors(voxel, &next_to);
    for (size_t other : next_to) {
      if (labels_[other] == kUnknown || labels_[other] == kOccupied) {
        labels_[voxel] = kContaminatedShadow;
        queue.emplace_back(voxel, 0);
        break;
      }
    }
  }
  while (!queue.empty()) {
    const auto [voxel, steps] = queue.front();
    queue.pop_front();
    if (steps >= max_steps) {
      continue;
    }
    next_to.clear();
    grid_.neighbors(voxel, &next_to);
    for (size_t other : next_to) {
      if (labels_[other] == kShadow) {
        labels_[other] = kContaminatedShadow;
        queue.emplace_back(other, steps + 1);
      }
    }
  }
}

void ObstacleGrid::update(const SelfModel & self)
{
  auto start = Clock::now();
  // The robot's body comes only from its shapes now: returns off the robot in an older
  // frame would leave a ghost arm where it was.
  self_flags_ = static_self_flags_;
  self.rasterize(grid_, params_.self_voxel_padding, &self_flags_);
  std::fill(occupied_by_.begin(), occupied_by_.end(), 0);
  for (int lidar = 0; lidar < lidar_count(); ++lidar) {
    for (size_t voxel : frames_[lidar].occupied) {
      occupied_by_[voxel] |= static_cast<uint8_t>(1u << lidar);
    }
  }
  timing_.self_voxels = ms_since(start);

  start = Clock::now();
  #pragma omp parallel for schedule(dynamic, 4096)
  for (size_t voxel = 0; voxel < labels_.size(); ++voxel) {
    labels_[voxel] = label_voxel(voxel);
  }
  timing_.label = ms_since(start);

  start = Clock::now();
  contaminate_shadows();
  timing_.contaminate = ms_since(start);

  start = Clock::now();
  occupied_table_.build(labels_, 1u << kOccupied, params_.table_threads);
  unsafe_table_.build(labels_, (1u << kUnknown) | (1u << kContaminatedShadow), params_.table_threads);
  timing_.tables = ms_since(start);
}

}  // namespace stretch_obstacle_grid
