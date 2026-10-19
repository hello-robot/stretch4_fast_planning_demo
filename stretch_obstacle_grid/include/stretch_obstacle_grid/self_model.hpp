// The robot's own body, as the planner's collision primitives posed in base_footprint.
//
// The caller poses the shapes for the current joint state (the planner's
// StretchCollisionModel.shape_poses does this), so this file has no kinematics. Points
// that fall inside a shape (plus padding) are the robot seeing itself, and voxels whose
// centers fall inside are labeled as the robot rather than as obstacles or shadow.

#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Geometry>

#include "stretch_obstacle_grid/cylindrical_grid.hpp"

namespace stretch_obstacle_grid
{

struct SelfShape
{
  enum class Type {kBox, kCylinder, kSphere, kCapsule};
  Type type;
  Eigen::Isometry3f pose;  // shape frame in base_footprint
  // kBox: half sizes along x, y, z. kCylinder and kCapsule: (radius, half length along z, -).
  // kSphere: (radius, -, -).
  Eigen::Vector3f dims;
};

class SelfModel
{
public:
  SelfModel() = default;
  explicit SelfModel(std::vector<SelfShape> shapes);

  const std::vector<SelfShape> & shapes() const {return shapes_;}

  // True if p (base_footprint) is inside any shape grown by `padding`.
  bool contains(const Eigen::Vector3f & p, float padding) const;

  // Set flags[i] = 1 for every voxel whose center is inside a shape grown by `padding`.
  void rasterize(const CylindricalGrid & grid, float padding, std::vector<uint8_t> * flags) const;

  // Distance along a ray (unit direction, base_footprint) to the first shape it enters from
  // outside, or infinity. Shapes within 5 cm of containing the origin don't count (the
  // LiDARs sit in the head). Capsules are treated as their bounding cylinders.
  float ray_distance(const Eigen::Vector3f & origin, const Eigen::Vector3f & direction) const;

private:
  // p in the shape's own frame.
  static bool shape_contains(const SelfShape & shape, const Eigen::Vector3f & p_in_shape, float padding);
  // Axis-aligned bounds of a padded shape in base_footprint.
  static Eigen::AlignedBox3f bounds(const SelfShape & shape, float padding);

  std::vector<SelfShape> shapes_;
  std::vector<Eigen::Isometry3f> shape_from_base_;
  std::vector<Eigen::AlignedBox3f> unpadded_bounds_;
  Eigen::AlignedBox3f all_bounds_;  // union of the unpadded bounds
};

}  // namespace stretch_obstacle_grid
