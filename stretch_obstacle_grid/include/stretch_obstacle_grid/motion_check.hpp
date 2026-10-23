// Checking a stretch of planned motion against the obstacle grid: how fast may the robot go?
//
// The caller (the trajectory server) poses the robot's shapes at samples along the motion
// ahead, in base_footprint as it is now, with base rotation already applied. Consecutive
// samples must be close: no point of the robot may move more than a few cm between them
// apart from base rotation, which may be anything under half a turn.
//
// Each shape is cut into pieces no bigger than max_piece on a side, and each piece's volume
// between two samples is bounded by a CylinderBox: the union of its bounds at both samples,
// grown by half the distance its corners moved (the most a point's straight path between
// samples can stray from both ends). Base rotation only shifts a piece in theta, so the union
// covers it exactly. The piece is also bounded by a box in x, y, z, since near the axis a
// CylinderBox grown by a few cm spans a full turn; a voxel counts only if both bounds hold it.
// A piece near the axis spans a wide theta range (a full turn if it surrounds the axis), which is
// why shapes are cut in every direction, not just along their length.
//
// For every piece that moves, the grid's box counts give its clearance to obstacles, to the
// ones that may move, and to unsafe space (unknown, contaminated shadow), and from those an
// allowed speed for that piece: speed and separation monitoring, where a moving obstacle is
// assumed to come toward the robot at obstacle_speed, a static one not at all, and unsafe
// space at unsafe_obstacle_speed. A step may take no less
// time than any piece's travel over its allowed speed.

#pragma once

#include <cstdint>
#include <vector>

#include "stretch_obstacle_grid/obstacle_grid.hpp"
#include "stretch_obstacle_grid/self_model.hpp"
#include "stretch_obstacle_grid/volume_table.hpp"

namespace stretch_obstacle_grid
{

struct MotionLimits
{
  // m/s at the contact point. Near obstacles, contact is allowed at contact_speed with
  // standoff 0; with a standoff the robot must be able to stop standoff short of them.
  float contact_speed = 0.10f;
  // m/s anywhere the robot sweeps unknown space or contaminated shadow, which might hide a
  // person.
  float unsafe_speed = 0.05f;
  float standoff = 0.0f;           // m (the "n" of the design)
  float obstacle_speed = 1.5f;     // m/s a moving (not static) obstacle is assumed to approach at
  // m/s something unseen is assumed to come out of unsafe space at. 0: the robot only has to be
  // down to unsafe_speed by the time it enters unsafe space.
  float unsafe_obstacle_speed = 0.0f;
  float latency = 0.25f;           // s from a frame's end to the robot starting to slow
  // m/s^2 any point can slow down at, when check() isn't given per-step braking (the arm's
  // max, the weakest joint; measured 2026-10-20)
  float deceleration = 0.7f;
  float max_speed = 1.0f;          // m/s, the most this ever allows (the joints cap it lower)
  float max_piece = 0.10f;         // m, largest side of the pieces a shape is cut into
  float max_clearance = 1.0f;      // m, clearances are searched up to this
  float clearance_resolution = 0.01f;  // m
  float min_motion = 0.001f;       // m: pieces moving less than this in a step aren't checked
};

// One piece's check over one step, for the slowest piece of the step.
struct StepLimit
{
  float min_duration = 0.0f;  // s: the step can't take less time than this
  float speed = 0.0f;         // m/s allowed for the limiting piece
  float travel = 0.0f;        // m the limiting piece moves (0: nothing moves)
  float clearance_occupied = 0.0f;  // m to any obstacle, smallest over the step's moving pieces
  float clearance_moving = 0.0f;    // m to obstacles that aren't static, likewise
  float clearance_unsafe = 0.0f;    // m, likewise
  // A moving piece sweeps an occupied (unsafe) voxel, or comes within the clearance search's
  // resolution of one.
  bool blocked = false;
  bool unsafe = false;
  int shape = -1;             // index of the limiting piece's shape
};

// What step_hits lists: the voxels behind a block (obstacles) or behind sweeping unsafe space.
enum class HitKind {kObstacles, kUnsafe};

// A voxel behind one step of a check.
struct StepHit
{
  size_t voxel;
  int shape;  // the first shape (index in the samples) whose swept piece holds it
  // True if both of the piece's bounds hold it. A piece counts as hitting when each bound holds
  // some flagged voxel, not necessarily the same one, so at the edge of an obstacle neither
  // may hold one the other does; then the cylindrical bound's voxels are listed, with false.
  bool in_both;
};

// Bounds of a convex set of points (base_footprint) in cylindrical coordinates. When the set
// surrounds the axis the theta range is a full turn and r_min is 0.
CylinderBox cylinder_bounds(const std::vector<Eigen::Vector3f> & points);

// The 8 corners (base_footprint) of the part of a shape between fractions f0 and f1 of its
// size along each of its axes (a cylinder, capsule or sphere is taken as its bounding box).
std::vector<Eigen::Vector3f> piece_corners(
  const SelfShape & shape, const Eigen::Vector3f & f0 = Eigen::Vector3f::Zero(),
  const Eigen::Vector3f & f1 = Eigen::Vector3f::Ones());

// A box grown by d on every side, in metric terms: every point within d of `box` is inside.
CylinderBox grow(const CylinderBox & box, float d);

// The smallest box covering both, with theta taken the short way around between them.
CylinderBox merge(const CylinderBox & a, const CylinderBox & b);

// The fastest a point may move with `clearance` (m) to something that may approach at
// `obstacle_speed` (m/s), if it must be down to `final_speed` (m/s) on arrival and can slow
// at `deceleration` (m/s^2).
float allowed_speed(
  float clearance, float final_speed, float obstacle_speed, float deceleration, const MotionLimits & limits);

class MotionCheck
{
public:
  explicit MotionCheck(const MotionLimits & limits = MotionLimits()) : limits_(limits) {}

  const MotionLimits & limits() const {return limits_;}

  // Limits for each step between consecutive samples. Every sample must have the same shapes
  // in the same order. `age` (s) is how old the oldest LiDAR frame behind the grid is (now
  // minus its end); obstacles may have come obstacle_speed * age closer since. `theta_slack`
  // (rad) widens every box in theta, for base rotation since that frame.
  //
  // `braking` (one per step, optional): max over the joints of |joint change over the step| /
  // that joint's deceleration (s^2). Slowing down along the path, every joint slows in
  // proportion, so the slowest-braking joint sets the stopping time, and a piece that moves
  // `travel` in the step can slow at travel / braking (m/s^2), whatever the speed. Without
  // it, every piece slows at limits.deceleration.
  std::vector<StepLimit> check(
    const ObstacleGrid & grid, const std::vector<SelfModel> & samples, float age = 0.0f,
    float theta_slack = 0.0f, const std::vector<float> & braking = {}) const;

  // The first step check(grid, samples, ..., theta_slack) would call blocked, or -1: the same
  // test, without the clearances and speeds, stopping at the first blocked step. For a planner
  // that only needs to know whether a motion is clear.
  int first_blocked(const ObstacleGrid & grid, const std::vector<SelfModel> & samples, float theta_slack = 0.0f) const;

  // The voxels behind step `step` of check(grid, samples, ..., theta_slack) being blocked
  // (kObstacles) or unsafe (kUnsafe): for each moving piece with clearance 0, the flagged
  // voxels both its bounds hold (see StepHit::in_both), grown by the smallest distance the
  // clearance search found blocked (0, or at most clearance_resolution). Empty exactly when
  // check() finds the step clear. Sorted by voxel, one entry each. For showing why a step was
  // limited.
  std::vector<StepHit> step_hits(
    const ObstacleGrid & grid, const std::vector<SelfModel> & samples, size_t step, HitKind kind,
    float theta_slack = 0.0f) const;

  // Clearance (m, a lower bound to clearance_resolution) from a box to the voxels a table
  // flags: 0 if the box holds one, max_clearance if none are that close.
  float clearance(const VolumeTable & table, const CylinderBox & box) const;

private:
  MotionLimits limits_;
};

}  // namespace stretch_obstacle_grid
