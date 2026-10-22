#include "stretch_obstacle_grid/self_model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace stretch_obstacle_grid
{

SelfModel::SelfModel(std::vector<SelfShape> shapes)
: shapes_(std::move(shapes))
{
  all_bounds_.setEmpty();
  for (const SelfShape & shape : shapes_) {
    shape_from_base_.push_back(shape.pose.inverse());
    unpadded_bounds_.push_back(bounds(shape, 0.0f));
    all_bounds_.extend(unpadded_bounds_.back());
  }
}

bool SelfModel::shape_contains(const SelfShape & shape, const Eigen::Vector3f & q, float padding)
{
  switch (shape.type) {
    case SelfShape::Type::kBox:
      return (q.cwiseAbs() - shape.dims).maxCoeff() <= padding;
    case SelfShape::Type::kCylinder:
      return q.head<2>().norm() <= shape.dims.x() + padding && std::abs(q.z()) <= shape.dims.y() + padding;
    case SelfShape::Type::kSphere:
      return q.norm() <= shape.dims.x() + padding;
    case SelfShape::Type::kCapsule: {
      const float z = std::clamp(q.z(), -shape.dims.y(), shape.dims.y());
      return (q - Eigen::Vector3f(0.0f, 0.0f, z)).norm() <= shape.dims.x() + padding;
    }
  }
  return false;
}

Eigen::AlignedBox3f SelfModel::bounds(const SelfShape & shape, float padding)
{
  Eigen::Vector3f half = Eigen::Vector3f::Zero();
  switch (shape.type) {
    case SelfShape::Type::kBox:
      half = shape.dims;
      break;
    case SelfShape::Type::kCylinder:
      half = Eigen::Vector3f(shape.dims.x(), shape.dims.x(), shape.dims.y());
      break;
    case SelfShape::Type::kSphere:
      half = Eigen::Vector3f::Constant(shape.dims.x());
      break;
    case SelfShape::Type::kCapsule:
      half = Eigen::Vector3f(shape.dims.x(), shape.dims.x(), shape.dims.y() + shape.dims.x());
      break;
  }
  // Half extents of the rotated box: |R| * half.
  const Eigen::Vector3f extent = shape.pose.linear().cwiseAbs() * half + Eigen::Vector3f::Constant(padding);
  const Eigen::Vector3f center = shape.pose.translation();
  return Eigen::AlignedBox3f(center - extent, center + extent);
}

namespace
{
constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kOriginClearance = 0.05f;  // m; see ray_distance

// Ray parameters [lo, hi] where a + b t + c t^2 <= 0 (c > 0), or an empty interval.
void quadratic_interval(float a, float b, float c, float * lo, float * hi)
{
  const float discriminant = b * b - 4.0f * a * c;
  if (discriminant < 0.0f) {
    *lo = kInf;
    *hi = -kInf;
    return;
  }
  const float root = std::sqrt(discriminant);
  *lo = (-b - root) / (2.0f * c);
  *hi = (-b + root) / (2.0f * c);
}

// Ray parameters inside the slab |o + t d| <= half along one axis.
void slab_interval(float o, float d, float half, float * lo, float * hi)
{
  if (std::abs(d) < 1e-9f) {
    *lo = std::abs(o) <= half ? -kInf : kInf;
    *hi = std::abs(o) <= half ? kInf : -kInf;
    return;
  }
  const float t0 = (-half - o) / d, t1 = (half - o) / d;
  *lo = std::min(t0, t1);
  *hi = std::max(t0, t1);
}
}  // namespace

float SelfModel::ray_distance(const Eigen::Vector3f & origin, const Eigen::Vector3f & direction, float padding) const
{
  // Most rays miss the robot entirely: check the bounds of all its shapes first.
  if (shapes_.empty()) {
    return kInf;
  }
  float lo_all = 0.0f, hi_all = kInf;
  for (int axis = 0; axis < 3; ++axis) {
    const float half = 0.5f * (all_bounds_.max()[axis] - all_bounds_.min()[axis]) + padding;
    const float center = 0.5f * (all_bounds_.max()[axis] + all_bounds_.min()[axis]);
    float a, b;
    slab_interval(origin[axis] - center, direction[axis], half, &a, &b);
    lo_all = std::max(lo_all, a);
    hi_all = std::min(hi_all, b);
  }
  if (lo_all > hi_all) {
    return kInf;
  }

  float nearest = kInf;
  for (size_t i = 0; i < shapes_.size(); ++i) {
    const SelfShape & shape = shapes_[i];
    const Eigen::Vector3f o = shape_from_base_[i] * origin;
    // A LiDAR sits inside the head (and anything drawn around it), or just outside its box
    // after leveling moves it a few cm: shapes that close to the ray's origin don't block it.
    if (shape_contains(shape, o, kOriginClearance)) {
      continue;
    }
    const Eigen::Vector3f d = shape_from_base_[i].linear() * direction;
    float lo = -kInf, hi = kInf, a, b;
    switch (shape.type) {
      case SelfShape::Type::kBox:
        for (int axis = 0; axis < 3; ++axis) {
          slab_interval(o[axis], d[axis], shape.dims[axis] + padding, &a, &b);
          lo = std::max(lo, a);
          hi = std::min(hi, b);
        }
        break;
      case SelfShape::Type::kCylinder:
      case SelfShape::Type::kCapsule: {
        const float half = shape.dims.y() + (shape.type == SelfShape::Type::kCapsule ? shape.dims.x() : 0.0f) + padding;
        slab_interval(o.z(), d.z(), half, &lo, &hi);
        const float c = d.x() * d.x() + d.y() * d.y();
        const float radius = shape.dims.x() + padding;
        const float a0 = o.x() * o.x() + o.y() * o.y() - radius * radius;
        if (c < 1e-12f) {
          if (a0 > 0.0f) {hi = -kInf;}  // parallel to the axis and outside it
        } else {
          quadratic_interval(a0, 2.0f * (o.x() * d.x() + o.y() * d.y()), c, &a, &b);
          lo = std::max(lo, a);
          hi = std::min(hi, b);
        }
        break;
      }
      case SelfShape::Type::kSphere:
        quadratic_interval(o.squaredNorm() - (shape.dims.x() + padding) * (shape.dims.x() + padding), 2.0f * o.dot(d), 1.0f,
          &lo, &hi);
        break;
    }
    if (lo <= hi && lo > 0.0f) {
      nearest = std::min(nearest, lo);
    }
  }
  return nearest;
}

bool SelfModel::contains(const Eigen::Vector3f & p, float padding) const
{
  if (shapes_.empty() || all_bounds_.exteriorDistance(p) > padding) {
    return false;
  }
  for (size_t i = 0; i < shapes_.size(); ++i) {
    if (unpadded_bounds_[i].exteriorDistance(p) <= padding &&
      shape_contains(shapes_[i], shape_from_base_[i] * p, padding))
    {
      return true;
    }
  }
  return false;
}

void SelfModel::rasterize(const CylindricalGrid & grid, float padding, std::vector<uint8_t> * flags) const
{
  const GridConfig & config = grid.config();
  for (size_t s = 0; s < shapes_.size(); ++s) {
    const SelfShape & shape = shapes_[s];
    const Eigen::AlignedBox3f box = bounds(shape, padding);

    // Cylindrical bounds of the shape's box: radius from the nearest and farthest points of
    // its footprint, theta from its corners unless the footprint contains the axis.
    const Eigen::Vector2f lo = box.min().head<2>(), hi = box.max().head<2>();
    const Eigen::Vector2f nearest = Eigen::Vector2f::Zero().cwiseMax(lo).cwiseMin(hi);
    const float r_near = nearest.norm();
    float r_far = 0.0f;
    std::vector<float> corner_thetas;
    for (float x : {lo.x(), hi.x()}) {
      for (float y : {lo.y(), hi.y()}) {
        r_far = std::max(r_far, std::hypot(x, y));
        corner_thetas.push_back(std::atan2(y, x));
      }
    }
    const bool around_axis = r_near == 0.0f;

    const int ri0 = std::max(0, static_cast<int>(std::floor((r_near - config.r_min) / config.cell)));
    const int ri1 = std::min(grid.radial_bins() - 1, static_cast<int>(std::floor((r_far - config.r_min) / config.cell)));
    const int zi0 = std::max(0, static_cast<int>(std::floor((box.min().z() - config.z_min) / config.cell)));
    const int zi1 = std::min(grid.z_bins() - 1, static_cast<int>(std::floor((box.max().z() - config.z_min) / config.cell)));
    if (ri0 > ri1 || zi0 > zi1) {
      continue;
    }

    // Theta span: the corners' angles, measured from the first one so a span crossing
    // +/-pi stays contiguous. The footprint doesn't contain the axis, so it spans < pi.
    float t_start = 0.0f, t_span = 2.0f * static_cast<float>(M_PI);
    if (!around_axis) {
      float min_offset = 0.0f, max_offset = 0.0f;
      for (float t : corner_thetas) {
        const float offset = std::remainder(t - corner_thetas[0], 2.0f * static_cast<float>(M_PI));
        min_offset = std::min(min_offset, offset);
        max_offset = std::max(max_offset, offset);
      }
      t_start = corner_thetas[0] + min_offset;
      t_span = max_offset - min_offset;
    }

    for (int ri = ri0; ri <= ri1; ++ri) {
      const Band & band = grid.band_of_radial(ri);
      const int first = CylindricalGrid::theta_bin(band, t_start);
      const int bins = std::min(band.theta_bins, static_cast<int>(t_span / band.dtheta) + 2);
      for (int k = 0; k < bins; ++k) {
        const int ti = (first + k) % band.theta_bins;
        for (int zi = zi0; zi <= zi1; ++zi) {
          const size_t index = grid.index(ri, ti, zi);
          float x, y, z;
          grid.center(index, &x, &y, &z);
          if (shape_contains(shape, shape_from_base_[s] * Eigen::Vector3f(x, y, z), padding)) {
            (*flags)[index] = 1;
          }
        }
      }
    }
  }
}

}  // namespace stretch_obstacle_grid
