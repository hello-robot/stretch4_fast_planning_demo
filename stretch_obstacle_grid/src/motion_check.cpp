#include "stretch_obstacle_grid/motion_check.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace stretch_obstacle_grid
{

namespace
{

constexpr float kPi = static_cast<float>(M_PI);
constexpr float kTwoPi = 2.0f * kPi;

CylinderBox full_turn(float r_min, float r_max, float z_min, float z_max)
{
  return CylinderBox{0.0f, kTwoPi, r_min, r_max, z_min, z_max};
}

bool is_full_turn(const CylinderBox & box)
{
  return box.theta_max - box.theta_min >= kTwoPi;
}

float cross(const Eigen::Vector2f & a, const Eigen::Vector2f & b)
{
  return a.x() * b.y() - a.y() * b.x();
}

// Convex hull (counterclockwise, no collinear points) of a few 2D points.
std::vector<Eigen::Vector2f> hull(std::vector<Eigen::Vector2f> p)
{
  std::sort(p.begin(), p.end(), [](const Eigen::Vector2f & a, const Eigen::Vector2f & b) {
      return a.x() < b.x() || (a.x() == b.x() && a.y() < b.y());
    });
  std::vector<Eigen::Vector2f> h(2 * p.size());
  size_t k = 0;
  for (size_t i = 0; i < p.size(); ++i) {  // lower hull
    while (k >= 2 && cross(h[k - 1] - h[k - 2], p[i] - h[k - 2]) <= 0.0f) {--k;}
    h[k++] = p[i];
  }
  for (size_t i = p.size() - 1, lower = k + 1; i-- > 0; ) {  // upper hull
    while (k >= lower && cross(h[k - 1] - h[k - 2], p[i] - h[k - 2]) <= 0.0f) {--k;}
    h[k++] = p[i];
  }
  h.resize(k > 1 ? k - 1 : k);
  return h;
}

float distance_to_segment(const Eigen::Vector2f & a, const Eigen::Vector2f & b)
{
  // From the origin.
  const Eigen::Vector2f ab = b - a;
  const float length2 = ab.squaredNorm();
  const float t = length2 > 0.0f ? std::clamp(-a.dot(ab) / length2, 0.0f, 1.0f) : 0.0f;
  return (a + t * ab).norm();
}

// Half sizes of a shape's bounding box in its own frame.
Eigen::Vector3f half_box(const SelfShape & shape)
{
  const Eigen::Vector3f & d = shape.dims;
  switch (shape.type) {
    case SelfShape::Type::kBox: return d;
    case SelfShape::Type::kCylinder: return Eigen::Vector3f(d[0], d[0], d[1]);
    case SelfShape::Type::kCapsule: return Eigen::Vector3f(d[0], d[0], d[1] + d[0]);
    case SelfShape::Type::kSphere: return Eigen::Vector3f(d[0], d[0], d[0]);
  }
  return d;
}

// One piece of a shape at one sample.
struct Piece
{
  std::vector<Eigen::Vector3f> corners;
  CylinderBox bounds;
};

CylinderBox widen_theta(CylinderBox box, float slack)
{
  if (slack <= 0.0f || is_full_turn(box)) {
    return box;
  }
  box.theta_min -= slack;
  box.theta_max += slack;
  return box.theta_max - box.theta_min >= kTwoPi ? full_turn(box.r_min, box.r_max, box.z_min, box.z_max) : box;
}

// A piece's volume over one step, bounded two ways, each holding everything within d of it.
// In cylindrical coordinates the bounds follow base rotation exactly, but grown by more than
// half the piece's distance from the axis they span a full turn; a box in x, y, z never does.
// Nothing flagged in either means nothing flagged within d.
struct SweptPiece
{
  CylinderBox cylinder;     // the bounds at both ends merged, grown by half the travel
  Eigen::AlignedBox3f box;  // the corners at both ends, grown by half the travel
  float theta_slack;        // rad of base rotation the grid may lag by

  bool clear(const VolumeTable & cylinder_table, const CartesianTable & box_table, float d) const
  {
    if (cylinder_table.count(widen_theta(grow(cylinder, d), theta_slack)) == 0) {
      return true;
    }
    // Turning by theta_slack moves a point at most its radius times that.
    const float x = std::max(std::abs(box.min().x()), std::abs(box.max().x()));
    const float y = std::max(std::abs(box.min().y()), std::abs(box.max().y()));
    const float g = d + std::hypot(x, y) * theta_slack;
    return box_table.count(Eigen::AlignedBox3f(box.min().array() - g, box.max().array() + g)) == 0;
  }
};

// The largest d (to resolution) for which clear(d) holds, by bisection: 0 if it fails at 0,
// max_clearance if it holds there. `known_clear`: a d already known to hold.
template<typename Clear>
float search_clearance(const Clear & clear, const MotionLimits & limits, float known_clear = 0.0f)
{
  if (known_clear <= 0.0f && !clear(0.0f)) {
    return 0.0f;
  }
  if (known_clear >= limits.max_clearance || clear(limits.max_clearance)) {
    return limits.max_clearance;
  }
  float lo = known_clear, hi = limits.max_clearance;
  while (hi - lo > limits.clearance_resolution) {
    const float middle = 0.5f * (lo + hi);
    (clear(middle) ? lo : hi) = middle;
  }
  return lo;
}

}  // namespace

CylinderBox cylinder_bounds(const std::vector<Eigen::Vector3f> & points)
{
  float z_min = std::numeric_limits<float>::infinity(), z_max = -z_min, r_max = 0.0f;
  std::vector<Eigen::Vector2f> xy;
  xy.reserve(points.size());
  for (const auto & p : points) {
    z_min = std::min(z_min, p.z());
    z_max = std::max(z_max, p.z());
    r_max = std::max(r_max, p.head<2>().norm());
    xy.push_back(p.head<2>());
  }

  // The axis is inside (or on) the footprint unless the origin is strictly outside an edge.
  const std::vector<Eigen::Vector2f> h = hull(xy);
  bool surrounds = h.size() >= 3;
  float r_min = std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < h.size(); ++i) {
    const Eigen::Vector2f & a = h[i];
    const Eigen::Vector2f & b = h[(i + 1) % h.size()];
    if (cross(b - a, -a) < 0.0f) {surrounds = false;}  // origin is right of this ccw edge
    r_min = std::min(r_min, distance_to_segment(a, b));
  }
  if (surrounds) {
    return full_turn(0.0f, r_max, z_min, z_max);
  }

  // Outside the footprint, the angles of its points span less than half a turn, so measure
  // them from one point's angle.
  const float reference = std::atan2(xy[0].y(), xy[0].x());
  float lo = 0.0f, hi = 0.0f;
  for (const auto & p : xy) {
    float d = std::atan2(p.y(), p.x()) - reference;
    d = d > kPi ? d - kTwoPi : (d < -kPi ? d + kTwoPi : d);
    lo = std::min(lo, d);
    hi = std::max(hi, d);
  }
  return CylinderBox{reference + lo, reference + hi, r_min, r_max, z_min, z_max};
}

std::vector<Eigen::Vector3f> piece_corners(
  const SelfShape & shape, const Eigen::Vector3f & f0, const Eigen::Vector3f & f1)
{
  const Eigen::Vector3f half = half_box(shape);
  const Eigen::Vector3f lo = -half + 2.0f * half.cwiseProduct(f0);
  const Eigen::Vector3f hi = -half + 2.0f * half.cwiseProduct(f1);
  std::vector<Eigen::Vector3f> corners;
  corners.reserve(8);
  for (int i = 0; i < 8; ++i) {
    const Eigen::Vector3f local((i & 1) ? hi.x() : lo.x(), (i & 2) ? hi.y() : lo.y(), (i & 4) ? hi.z() : lo.z());
    corners.push_back(shape.pose * local);
  }
  return corners;
}

CylinderBox grow(const CylinderBox & box, float d)
{
  const float r_min = std::max(0.0f, box.r_min - d);
  CylinderBox out{box.theta_min, box.theta_max, r_min, box.r_max + d, box.z_min - d, box.z_max + d};
  if (is_full_turn(box) || d <= 0.0f) {
    return out;
  }
  // A point within d of the box, at radius r >= r_min, is at most asin(d / r) outside its
  // theta range; nearer the axis than d it could be at any angle.
  if (r_min <= d) {
    return full_turn(out.r_min, out.r_max, out.z_min, out.z_max);
  }
  const float dtheta = std::asin(d / r_min);
  out.theta_min -= dtheta;
  out.theta_max += dtheta;
  if (out.theta_max - out.theta_min >= kTwoPi) {
    return full_turn(out.r_min, out.r_max, out.z_min, out.z_max);
  }
  return out;
}

CylinderBox merge(const CylinderBox & a, const CylinderBox & b)
{
  const float r_min = std::min(a.r_min, b.r_min), r_max = std::max(a.r_max, b.r_max);
  const float z_min = std::min(a.z_min, b.z_min), z_max = std::max(a.z_max, b.z_max);
  if (is_full_turn(a) || is_full_turn(b)) {
    return full_turn(r_min, r_max, z_min, z_max);
  }
  // Shift b by whole turns so its middle is within half a turn of a's.
  const float middle_a = 0.5f * (a.theta_min + a.theta_max);
  const float middle_b = 0.5f * (b.theta_min + b.theta_max);
  const float shift = kTwoPi * std::round((middle_a - middle_b) / kTwoPi);
  const float lo = std::min(a.theta_min, b.theta_min + shift);
  const float hi = std::max(a.theta_max, b.theta_max + shift);
  if (hi - lo >= kTwoPi) {
    return full_turn(r_min, r_max, z_min, z_max);
  }
  return CylinderBox{lo, hi, r_min, r_max, z_min, z_max};
}

float allowed_speed(
  float clearance, float final_speed, float obstacle_speed, float deceleration, const MotionLimits & limits)
{
  // Moving at v when an obstacle shows up `clearance` away and closes at obstacle_speed, the
  // robot keeps going at v for the latency, then slows at `deceleration` to final_speed. The
  // gap must last until then:
  //   clearance >= (v_o + v) t_l + v_o (v - v_f) / a + (v^2 - v_f^2) / (2a)
  // Solve the quadratic in v for equality.
  const float a = deceleration, t = limits.latency, vo = obstacle_speed, vf = final_speed;
  const float qa = 0.5f / a;
  const float qb = t + vo / a;
  const float qc = vo * t - vo * vf / a - vf * vf * qa - clearance;
  const float v = (-qb + std::sqrt(std::max(0.0f, qb * qb - 4.0f * qa * qc))) / (2.0f * qa);
  return std::clamp(v, vf, limits.max_speed);
}

float MotionCheck::clearance(const VolumeTable & table, const CylinderBox & box) const
{
  return search_clearance([&](float d) {return table.count(grow(box, d)) == 0;}, limits_);
}

std::vector<StepLimit> MotionCheck::check(
  const ObstacleGrid & grid, const std::vector<SelfModel> & samples, float age, float theta_slack,
  const std::vector<float> & braking) const
{
  if (!braking.empty() && braking.size() + 1 != samples.size()) {
    throw std::invalid_argument("MotionCheck::check: braking needs one value per step");
  }
  std::vector<StepLimit> steps;
  if (samples.size() < 2) {
    return steps;
  }
  const size_t shape_count = samples[0].shapes().size();
  for (const auto & sample : samples) {
    if (sample.shapes().size() != shape_count) {
      throw std::invalid_argument("MotionCheck::check: every sample needs the same shapes");
    }
  }

  // Cut each shape into the same pieces at every sample (the arm box changes length): along
  // each of its axes, into as many as its longest size over the samples needs. Then bound
  // every piece at every sample.
  std::vector<Eigen::Vector3i> cuts(shape_count);
  std::vector<int> piece_count(shape_count);
  for (size_t s = 0; s < shape_count; ++s) {
    Eigen::Vector3f longest = Eigen::Vector3f::Zero();
    for (const auto & sample : samples) {
      longest = longest.cwiseMax(2.0f * half_box(sample.shapes()[s]));
    }
    for (int i = 0; i < 3; ++i) {
      cuts[s][i] = std::max(1, static_cast<int>(std::ceil(longest[i] / limits_.max_piece)));
    }
    piece_count[s] = cuts[s].prod();
  }
  std::vector<std::vector<Piece>> pieces(samples.size());  // [sample][piece]
  for (size_t k = 0; k < samples.size(); ++k) {
    for (size_t s = 0; s < shape_count; ++s) {
      const SelfShape & shape = samples[k].shapes()[s];
      const Eigen::Vector3f step = cuts[s].cast<float>().cwiseInverse();
      for (int p = 0; p < piece_count[s]; ++p) {
        const Eigen::Vector3i at(p % cuts[s][0], (p / cuts[s][0]) % cuts[s][1], p / (cuts[s][0] * cuts[s][1]));
        const Eigen::Vector3f f0 = at.cast<float>().cwiseProduct(step);
        Piece piece;
        piece.corners = piece_corners(shape, f0, f0 + step);
        piece.bounds = cylinder_bounds(piece.corners);
        pieces[k].push_back(std::move(piece));
      }
    }
  }

  // Moving obstacles (and anything in unsafe space) may have come this much closer since the
  // oldest frame. Static ones stay put.
  const float approach = limits_.obstacle_speed * std::max(0.0f, age);
  const float unsafe_approach = limits_.unsafe_obstacle_speed * std::max(0.0f, age);
  const float occupied_final = limits_.standoff > 0.0f ? 0.0f : limits_.contact_speed;

  steps.resize(samples.size() - 1);
  for (size_t k = 0; k + 1 < samples.size(); ++k) {
    StepLimit & step = steps[k];
    step.clearance_occupied = step.clearance_moving = step.clearance_unsafe = limits_.max_clearance;
    size_t piece_index = 0;
    for (size_t s = 0; s < shape_count; ++s) {
      for (int p = 0; p < piece_count[s]; ++p, ++piece_index) {
        const Piece & a = pieces[k][piece_index];
        const Piece & b = pieces[k + 1][piece_index];
        // A rigid (or stretching) motion moves every point of a box no more than its corners.
        float travel = 0.0f;
        for (size_t c = 0; c < a.corners.size(); ++c) {
          travel = std::max(travel, (b.corners[c] - a.corners[c]).norm());
        }
        if (travel < limits_.min_motion) {
          continue;
        }

        SweptPiece swept{grow(merge(a.bounds, b.bounds), 0.5f * travel), Eigen::AlignedBox3f(), theta_slack};
        for (size_t c = 0; c < a.corners.size(); ++c) {
          swept.box.extend(a.corners[c]);
          swept.box.extend(b.corners[c]);
        }
        swept.box.min().array() -= 0.5f * travel;
        swept.box.max().array() += 0.5f * travel;
        const float to_occupied = search_clearance([&](float d) {
              return swept.clear(grid.occupied_table(), grid.occupied_box_table(), d);
            }, limits_);
        // Moving obstacles are a subset of all of them, so at least as far away.
        const float to_moving = search_clearance([&](float d) {
              return swept.clear(grid.moving_table(), grid.moving_box_table(), d);
            }, limits_, to_occupied);
        const float to_unsafe = search_clearance([&](float d) {
              return swept.clear(grid.unsafe_table(), grid.unsafe_box_table(), d);
            }, limits_);
        step.blocked |= to_occupied == 0.0f;
        step.unsafe |= to_unsafe == 0.0f;
        step.clearance_occupied = std::min(step.clearance_occupied, to_occupied);
        step.clearance_moving = std::min(step.clearance_moving, to_moving);
        step.clearance_unsafe = std::min(step.clearance_unsafe, to_unsafe);

        const float decel = braking.empty() || braking[k] <= 0.0f ? limits_.deceleration : travel / braking[k];
        const float speed = std::min({
          allowed_speed(to_occupied - limits_.standoff, occupied_final, 0.0f, decel, limits_),
          allowed_speed(to_moving - approach - limits_.standoff, occupied_final, limits_.obstacle_speed, decel, limits_),
          allowed_speed(to_unsafe - unsafe_approach, limits_.unsafe_speed, limits_.unsafe_obstacle_speed, decel, limits_)});
        const float duration = speed > 0.0f ? travel / speed : std::numeric_limits<float>::infinity();
        if (duration > step.min_duration || step.shape < 0) {
          step.min_duration = duration;
          step.speed = speed;
          step.travel = travel;
          step.shape = static_cast<int>(s);
        }
      }
    }
  }
  return steps;
}

}  // namespace stretch_obstacle_grid
