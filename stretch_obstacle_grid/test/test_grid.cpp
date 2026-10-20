// Tests for the grid layout, the box-count tables, the robot's shapes, labeling a synthetic
// LiDAR scene, and checking motion against it.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <random>

#include "stretch_obstacle_grid/motion_check.hpp"
#include "stretch_obstacle_grid/obstacle_grid.hpp"

using namespace stretch_obstacle_grid;

namespace stretch_obstacle_grid
{
// So failed expectations print label names.
void PrintTo(Label label, std::ostream * os)
{
  static const char * names[] = {"free", "occupied", "self", "shadow", "contaminated shadow", "unknown"};
  *os << names[label];
}
}  // namespace stretch_obstacle_grid

namespace
{
constexpr float kPi = static_cast<float>(M_PI);
constexpr float kTwoPi = 2.0f * kPi;

GridConfig small_config()
{
  GridConfig config;
  config.r_max = 0.9f;
  config.z_max = 0.3f;
  return config;
}
}  // namespace

// --- Layout -------------------------------------------------------------------------------

TEST(CylindricalGrid, BandsDoubleOutward)
{
  const CylindricalGrid grid;
  std::vector<int> bins;
  std::vector<float> starts;
  for (const Band & band : grid.bands()) {
    bins.push_back(band.theta_bins);
    starts.push_back(grid.config().r_min + band.r_begin * grid.config().cell);
  }
  EXPECT_EQ(bins, (std::vector<int>{8, 16, 32, 64, 128, 256}));
  EXPECT_NEAR(starts[1], 0.04f, 1e-5);
  EXPECT_NEAR(starts[2], 0.10f, 1e-5);
  EXPECT_NEAR(starts[3], 0.20f, 1e-5);
  EXPECT_NEAR(starts[4], 0.40f, 1e-5);
  EXPECT_NEAR(starts[5], 0.80f, 1e-5);
  EXPECT_EQ(grid.radial_bins(), 79);
  EXPECT_EQ(grid.z_bins(), 98);

  // Every voxel's arc is at most two cells, and at least about one. Within 10 cm of the
  // axis, where 2 cm radial bins are coarse next to the radius, arcs go down to 0.75 cells.
  const float cell = grid.config().cell;
  for (int ri = 0; ri < grid.radial_bins(); ++ri) {
    const Band & band = grid.band_of_radial(ri);
    const float inner = grid.config().r_min + ri * cell;
    EXPECT_GE(inner * band.dtheta, (inner < 0.1f ? 0.75f : 0.95f) * cell) << "radial bin " << ri;
    EXPECT_LE((inner + cell) * band.dtheta, 2.0f * cell + 1e-5f) << "radial bin " << ri;
  }
}

TEST(CylindricalGrid, IndexCellAndLocateAgree)
{
  const CylindricalGrid grid;
  std::mt19937 rng(1);
  std::uniform_int_distribution<size_t> pick(0, grid.size() - 1);
  for (int k = 0; k < 20000; ++k) {
    const size_t index = pick(rng);
    const Cell c = grid.cell(index);
    EXPECT_EQ(grid.index(c.ri, c.ti, c.zi), index);
    float x, y, z;
    grid.center(index, &x, &y, &z);
    size_t located;
    ASSERT_TRUE(grid.locate(x, y, z, &located));
    EXPECT_EQ(located, index);
  }
  size_t unused;
  EXPECT_FALSE(grid.locate(0.01f, 0.0f, 1.0f, &unused));  // inside r_min
  EXPECT_FALSE(grid.locate(1.7f, 0.0f, 1.0f, &unused));   // past r_max
  EXPECT_FALSE(grid.locate(1.0f, 0.0f, 0.01f, &unused));  // below z_min
}

TEST(CylindricalGrid, NeighborsAreMutual)
{
  const CylindricalGrid grid;
  std::mt19937 rng(2);
  std::uniform_int_distribution<size_t> pick(0, grid.size() - 1);
  std::vector<size_t> mine, theirs;
  for (int k = 0; k < 5000; ++k) {
    const size_t index = pick(rng);
    mine.clear();
    grid.neighbors(index, &mine);
    for (size_t other : mine) {
      theirs.clear();
      grid.neighbors(other, &theirs);
      EXPECT_NE(std::find(theirs.begin(), theirs.end(), index), theirs.end())
        << index << " lists " << other << " but not the reverse";
    }
  }
}

// --- Box counts ---------------------------------------------------------------------------

// Whether a voxel overlaps a box, straight from the definitions.
bool overlaps(const CylindricalGrid & grid, size_t index, const CylinderBox & box)
{
  float r0, r1, t0, t1, z0, z1;
  grid.bounds(grid.cell(index), &r0, &r1, &t0, &t1, &z0, &z1);
  if (r0 > box.r_max || r1 < box.r_min || z0 > box.z_max || z1 < box.z_min) {
    return false;
  }
  const float span = box.theta_max - box.theta_min;
  if (span >= kTwoPi) {
    return true;
  }
  float start = std::fmod(t0 - box.theta_min, kTwoPi);  // the voxel's start, relative to the box's
  if (start < 0.0f) {start += kTwoPi;}
  return start <= span || start + (t1 - t0) >= kTwoPi;
}

TEST(VolumeTable, CountsMatchBruteForce)
{
  const CylindricalGrid grid(small_config());
  std::mt19937 rng(3);
  std::bernoulli_distribution coin(0.1);
  std::vector<uint8_t> flags(grid.size());
  for (auto & f : flags) {f = coin(rng);}
  VolumeTable table(grid);
  table.build(flags, 1u << 1);

  std::uniform_real_distribution<float> unit(0.0f, 1.0f);
  for (int k = 0; k < 300; ++k) {
    const float theta_min = (unit(rng) - 0.5f) * 4.0f * kPi;  // also tests wrapping
    const float span = unit(rng) < 0.1f ? 7.0f : unit(rng) * 3.0f;
    const float r_a = unit(rng), r_b = unit(rng), z_a = unit(rng) * 0.4f, z_b = unit(rng) * 0.4f;
    const CylinderBox box{theta_min, theta_min + span, std::min(r_a, r_b), std::max(r_a, r_b),
      std::min(z_a, z_b), std::max(z_a, z_b)};
    uint32_t expected = 0;
    for (size_t i = 0; i < grid.size(); ++i) {
      expected += flags[i] && overlaps(grid, i, box);
    }
    EXPECT_EQ(table.count(box), expected) << "box " << k;
  }
}

// --- Robot shapes ---------------------------------------------------------------------------

TEST(CartesianTable, CountsEveryOverlappingVoxel)
{
  const CylindricalGrid grid(small_config());
  std::mt19937 rng(7);
  std::vector<uint8_t> values(grid.size());
  std::bernoulli_distribution flagged(0.02);
  for (auto & v : values) {v = flagged(rng) ? 1 : 0;}
  CartesianTable table(grid);
  table.build(values, 1u << 1);

  std::uniform_real_distribution<float> coordinate(-0.9f, 0.9f), height(0.0f, 0.3f), size(0.0f, 0.2f);
  for (int trial = 0; trial < 200; ++trial) {
    const Eigen::Vector3f lo(coordinate(rng), coordinate(rng), height(rng));
    const Eigen::AlignedBox3f box(lo, lo + Eigen::Vector3f(size(rng), size(rng), size(rng)));
    // Flagged voxels with a corner or center inside the box must be counted; everything counted
    // must be within a cell and a voxel of it.
    uint32_t must = 0, may = 0;
    const Eigen::AlignedBox3f near(box.min().array() - table.cell() - table.voxel_reach(),
      box.max().array() + table.cell() + table.voxel_reach());
    for (size_t i = 0; i < grid.size(); ++i) {
      if (!values[i]) {continue;}
      Eigen::Vector3f c;
      grid.center(i, &c.x(), &c.y(), &c.z());
      float r0, r1, t0, t1, z0, z1;
      grid.bounds(grid.cell(i), &r0, &r1, &t0, &t1, &z0, &z1);
      bool touches = box.contains(c);
      for (float r : {r0, r1}) {
        for (float t : {t0, t1}) {
          for (float z : {z0, z1}) {
            touches |= box.contains(Eigen::Vector3f(r * std::cos(t), r * std::sin(t), z));
          }
        }
      }
      must += touches;
      may += near.contains(c);
    }
    const uint32_t counted = table.count(box);
    EXPECT_GE(counted, must);
    EXPECT_LE(counted, may);
  }
}

TEST(SelfModel, ContainsAndRasterizes)
{
  Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
  pose.translate(Eigen::Vector3f(0.5f, 0.2f, 0.15f));
  pose.rotate(Eigen::AngleAxisf(0.3f, Eigen::Vector3f::UnitZ()));
  const SelfModel self({SelfShape{SelfShape::Type::kBox, pose, Eigen::Vector3f(0.1f, 0.05f, 0.05f)}});

  EXPECT_TRUE(self.contains(Eigen::Vector3f(0.5f, 0.2f, 0.15f), 0.0f));
  EXPECT_FALSE(self.contains(Eigen::Vector3f(0.5f, 0.2f, 0.22f), 0.0f));
  EXPECT_TRUE(self.contains(Eigen::Vector3f(0.5f, 0.2f, 0.22f), 0.03f));

  const CylindricalGrid grid(small_config());
  std::vector<uint8_t> flags(grid.size());
  self.rasterize(grid, 0.0f, &flags);
  size_t marked = 0;
  for (size_t i = 0; i < grid.size(); ++i) {
    float x, y, z;
    grid.center(i, &x, &y, &z);
    EXPECT_EQ(flags[i] != 0, self.contains(Eigen::Vector3f(x, y, z), 0.0f)) << "voxel " << i;
    marked += flags[i];
  }
  // The box is 0.2 x 0.1 x 0.1 m = 2 liters, about 250 voxels of 8 cm^3.
  EXPECT_GT(marked, 150u);
  EXPECT_LT(marked, 400u);
}

// --- A synthetic LiDAR scene -----------------------------------------------------------------

// The scene's grid reaches 2 m (past the default), so there's room for a wall inside it.
GridConfig scene_config()
{
  GridConfig config;
  config.r_max = 2.0f;
  return config;
}

// A LiDAR 1 m up on the axis, level, with rings every degree from -60 to +30, looking at:
// the floor, a cylindrical wall at `wall_radius` around the axis, the robot's "arm" (a box
// just below and in front of the LiDAR) and optional obstacle boxes.
struct Scene
{
  float wall_radius = 2.5f;
  Eigen::AlignedBox3f arm{Eigen::Vector3f(0.3f, -0.05f, 0.85f), Eigen::Vector3f(0.5f, 0.05f, 0.95f)};
  std::vector<Eigen::AlignedBox3f> obstacles;

  static constexpr int kAzimuthBins = 720;
  static constexpr float kHeight = 1.0f;

  static std::vector<float> rings()
  {
    std::vector<float> elevation;
    for (int deg = -60; deg <= 30; ++deg) {elevation.push_back(deg * kPi / 180.0f);}
    return elevation;
  }

  LidarModel lidar() const
  {
    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
    pose.translate(Eigen::Vector3f(0.0f, 0.0f, kHeight));
    return LidarModel{pose, rings(), kAzimuthBins};
  }

  SelfModel self() const
  {
    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
    pose.translate(arm.center());
    return SelfModel({SelfShape{SelfShape::Type::kBox, pose, arm.sizes() / 2.0f}});
  }

  // Distance along a unit ray from `origin` to a box, if it hits.
  static std::optional<float> hit_box(const Eigen::Vector3f & origin, const Eigen::Vector3f & dir,
    const Eigen::AlignedBox3f & box)
  {
    float near = 0.0f, far = std::numeric_limits<float>::infinity();
    for (int i = 0; i < 3; ++i) {
      if (std::abs(dir[i]) < 1e-9f) {
        if (origin[i] < box.min()[i] || origin[i] > box.max()[i]) {return std::nullopt;}
        continue;
      }
      float a = (box.min()[i] - origin[i]) / dir[i], b = (box.max()[i] - origin[i]) / dir[i];
      if (a > b) {std::swap(a, b);}
      near = std::max(near, a);
      far = std::min(far, b);
    }
    return near <= far ? std::optional<float>(near) : std::nullopt;
  }

  // Points in the LiDAR's frame (NaN for no return) and their rings.
  void cast(std::vector<float> * xyz, std::vector<uint16_t> * ring) const
  {
    const std::vector<float> elevation = rings();
    const Eigen::Vector3f origin(0.0f, 0.0f, kHeight);
    for (int col = 0; col < kAzimuthBins; ++col) {
      const float azimuth = -kPi + (col + 0.5f) * kTwoPi / kAzimuthBins;
      for (size_t r = 0; r < elevation.size(); ++r) {
        const Eigen::Vector3f dir(std::cos(elevation[r]) * std::cos(azimuth),
          std::cos(elevation[r]) * std::sin(azimuth), std::sin(elevation[r]));
        float best = std::numeric_limits<float>::infinity();
        if (dir.z() < 0.0f) {best = std::min(best, -kHeight / dir.z());}       // floor
        best = std::min(best, wall_radius / dir.head<2>().norm());              // wall
        for (const auto & box : obstacles) {
          if (auto t = hit_box(origin, dir, box)) {best = std::min(best, *t);}
        }
        if (auto t = hit_box(origin, dir, arm)) {best = std::min(best, *t);}
        const Eigen::Vector3f p = dir * best;
        xyz->insert(xyz->end(), {p.x(), p.y(), p.z()});
        ring->push_back(static_cast<uint16_t>(r));
      }
    }
  }
};

Label label_at(const ObstacleGrid & og, float x, float y, float z)
{
  size_t voxel;
  EXPECT_TRUE(og.grid().locate(x, y, z, &voxel));
  return static_cast<Label>(og.labels()[voxel]);
}

std::unique_ptr<ObstacleGrid> label_scene(
  const Scene & scene, const ObstacleGridParams & params = ObstacleGridParams())
{
  auto og = std::make_unique<ObstacleGrid>(scene_config(), std::vector<LidarModel>{scene.lidar()}, params);
  std::vector<float> xyz;
  std::vector<uint16_t> ring;
  scene.cast(&xyz, &ring);
  const SelfModel self = scene.self();
  og->add_frame(0, xyz.data(), ring.data(), ring.size(), 0.0, self);
  og->update(self);
  return og;
}

TEST(ObstacleGrid, ArmShadowAloneIsClean)
{
  Scene scene;
  // Without its own shapes the arm would just be an obstacle; label_scene passes them.
  const auto labeled = label_scene(scene);
  const ObstacleGrid & og = *labeled;
  EXPECT_EQ(label_at(og, 1.0f, 0.0f, 1.5f), kFree);    // above everything
  EXPECT_EQ(label_at(og, 1.0f, 0.0f, 0.3f), kFree);    // under the arm's shadow
  EXPECT_EQ(label_at(og, 0.4f, 0.0f, 0.9f), kSelf);    // the arm itself
  // Straight back from the LiDAR through the arm's middle: (0.4, 0, 0.9) -> (0.8, 0, 0.8).
  EXPECT_EQ(label_at(og, 0.8f, 0.0f, 0.8f), kShadow);
  EXPECT_EQ(label_at(og, 1.6f, 0.0f, 0.6f), kShadow);
}

TEST(ObstacleGrid, UnknownSpaceContaminatesShadowOnlyIfAskedTo)
{
  Scene scene;
  scene.wall_radius = 1.5f;  // inside the grid: the shadow runs into the unseen wall behind it
  const auto labeled = label_scene(scene);
  const ObstacleGrid & og = *labeled;
  EXPECT_EQ(label_at(og, 1.51f, 0.0f, 1.2f), kOccupied);  // the wall
  EXPECT_EQ(label_at(og, 1.49f, 0.0f, 1.2f), kUnknown);   // within range noise in front of it
  EXPECT_EQ(label_at(og, 1.8f, 0.0f, 1.2f), kUnknown);    // behind the wall
  // The shadow reaches the wall where the arm hides it, so that part of the wall is unknown,
  // not an obstacle: by default the shadow next to it stays clean.
  EXPECT_EQ(label_at(og, 1.4f, 0.0f, 0.6f), kShadow);

  // With unknown space contaminating too, the shadow next to the wall is contaminated; 0.6 m
  // back toward the arm it isn't.
  ObstacleGridParams params;
  params.unknown_contaminates_shadow = true;
  const auto strict = label_scene(scene, params);
  EXPECT_EQ(label_at(*strict, 1.4f, 0.0f, 0.6f), kContaminatedShadow);
  EXPECT_EQ(label_at(*strict, 0.8f, 0.0f, 0.8f), kShadow);
}

TEST(ObstacleGrid, NearTheLidarIsTheRobotsBlindSpot)
{
  Scene scene;
  const auto labeled = label_scene(scene);
  // 0.2 m above the LiDAR: nothing there returns (and the rings only go to +30 deg), so no
  // LiDAR sees it, but it's within self_blind_range.
  EXPECT_EQ(label_at(*labeled, 0.0f, 0.1f, Scene::kHeight + 0.2f), kShadow);
  EXPECT_EQ(label_at(*labeled, 0.0f, 0.1f, Scene::kHeight + 0.6f), kUnknown);
}

TEST(ObstacleGrid, UnlimitedReachContaminatesWholeShadow)
{
  Scene scene;
  scene.wall_radius = 1.5f;
  ObstacleGridParams params;
  params.contamination_reach = std::numeric_limits<float>::infinity();
  const auto labeled = label_scene(scene, params);
  EXPECT_EQ(label_at(*labeled, 0.8f, 0.0f, 0.8f), kContaminatedShadow);
}

TEST(ObstacleGrid, ObstacleTouchingShadowContaminatesIt)
{
  Scene scene;
  // The shadow's lower edge is the ray past the arm's front-bottom corner (0.3, 0.85):
  // z = 1 - 0.5 x, so z = 0.5 at x = 1. This box's top just meets it there.
  scene.obstacles.emplace_back(Eigen::Vector3f(0.9f, -0.05f, 0.4f), Eigen::Vector3f(1.0f, 0.05f, 0.5f));
  const auto labeled = label_scene(scene);
  const ObstacleGrid & og = *labeled;
  EXPECT_EQ(label_at(og, 0.95f, 0.0f, 0.57f), kContaminatedShadow);  // just above the box
  EXPECT_EQ(label_at(og, 0.6f, 0.0f, 0.85f), kShadow);               // about 0.5 m away

  // The box counts see the obstacle and the space behind it.
  EXPECT_GT(og.count_occupied(CylinderBox{-0.1f, 0.1f, 0.85f, 1.05f, 0.35f, 0.55f}), 0u);
  EXPECT_EQ(og.count_occupied(CylinderBox{2.0f, 2.5f, 0.85f, 1.05f, 0.35f, 0.55f}), 0u);
  EXPECT_GT(og.count_unsafe(CylinderBox{-0.05f, 0.05f, 1.1f, 1.3f, 0.25f, 0.35f}), 0u);
}

TEST(ObstacleGrid, ObstacleInsideShadowIsUnseen)
{
  // The case the shadow rule accepts: something entirely under the arm can't be seen.
  Scene scene;
  scene.obstacles.emplace_back(Eigen::Vector3f(0.9f, -0.02f, 0.6f), Eigen::Vector3f(0.95f, 0.02f, 0.7f));
  const auto labeled = label_scene(scene);
  const ObstacleGrid & og = *labeled;
  EXPECT_EQ(og.count_occupied(CylinderBox{-0.1f, 0.1f, 0.85f, 1.0f, 0.55f, 0.75f}), 0u);
  EXPECT_EQ(label_at(og, 0.8f, 0.0f, 0.8f), kShadow);
}

TEST(ObstacleGrid, HiddenByObstacleIsUnknown)
{
  Scene scene;
  scene.obstacles.emplace_back(Eigen::Vector3f(0.8f, 0.6f, 0.4f), Eigen::Vector3f(0.9f, 0.7f, 0.5f));
  const auto labeled = label_scene(scene);
  const ObstacleGrid & og = *labeled;
  // Behind the box along the ray from (0, 0, 1) through its center (0.85, 0.65, 0.45).
  EXPECT_EQ(label_at(og, 1.1f, 0.84f, 0.29f), kUnknown);
  EXPECT_EQ(label_at(og, 0.85f, 0.4f, 0.45f), kFree);
}

TEST(SelfModel, RayDistance)
{
  Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
  pose.translate(Eigen::Vector3f(1.0f, 0.0f, 0.0f));
  Eigen::Isometry3f around_origin = Eigen::Isometry3f::Identity();
  const SelfModel self({
      SelfShape{SelfShape::Type::kBox, pose, Eigen::Vector3f(0.1f, 0.1f, 0.1f)},
      SelfShape{SelfShape::Type::kCylinder, pose, Eigen::Vector3f(0.1f, 0.1f, 0.0f)},
      SelfShape{SelfShape::Type::kSphere, around_origin, Eigen::Vector3f(0.5f, 0.0f, 0.0f)}});
  // The sphere contains the origin, so it doesn't count; the box is entered at 0.9.
  EXPECT_NEAR(self.ray_distance(Eigen::Vector3f::Zero(), Eigen::Vector3f::UnitX()), 0.9f, 1e-5);
  EXPECT_TRUE(std::isinf(self.ray_distance(Eigen::Vector3f::Zero(), Eigen::Vector3f::UnitY())));
  EXPECT_TRUE(std::isinf(self.ray_distance(Eigen::Vector3f::Zero(), -Eigen::Vector3f::UnitX())));
}

TEST(ObstacleGrid, StaticSelfMatchesPerFrameSelf)
{
  Scene scene;
  scene.wall_radius = 1.5f;
  const auto per_frame = label_scene(scene);

  // The same arm, given once as a static part, with nothing passed per frame.
  ObstacleGrid og(scene_config(), {scene.lidar()});
  og.set_static_self(scene.self());
  std::vector<float> xyz;
  std::vector<uint16_t> ring;
  scene.cast(&xyz, &ring);
  const SelfModel nothing;
  og.add_frame(0, xyz.data(), ring.data(), ring.size(), 0.0, nothing);
  og.update(nothing);
  EXPECT_EQ(og.labels(), per_frame->labels());
}

TEST(ObstacleGrid, DuplicateReturnsAreSkipped)
{
  Scene scene;
  scene.wall_radius = 1.5f;
  const auto single = label_scene(scene);

  // Dual-return layout: each firing's points twice in a row.
  std::vector<float> xyz, dual_xyz;
  std::vector<uint16_t> ring, dual_ring;
  scene.cast(&xyz, &ring);
  const size_t rings = Scene::rings().size();
  for (size_t start = 0; start < ring.size(); start += rings) {
    for (int copy = 0; copy < 2; ++copy) {
      dual_xyz.insert(dual_xyz.end(), xyz.begin() + 3 * start, xyz.begin() + 3 * (start + rings));
      dual_ring.insert(dual_ring.end(), ring.begin() + start, ring.begin() + start + rings);
    }
  }
  ObstacleGridParams params;
  params.duplicate_stride = static_cast<int>(rings);
  ObstacleGrid og(scene_config(), {scene.lidar()}, params);
  const SelfModel self = scene.self();
  og.add_frame(0, dual_xyz.data(), dual_ring.data(), dual_ring.size(), 0.0, self);
  og.update(self);
  EXPECT_EQ(og.labels(), single->labels());
}

TEST(ObstacleGrid, IsolatedReturnIsIgnored)
{
  Scene scene;
  std::vector<float> xyz;
  std::vector<uint16_t> ring;
  scene.cast(&xyz, &ring);
  // One stray return floating in free space, 1 m out to the side (away from the arm).
  const std::vector<float> elevation = Scene::rings();
  const int r = 40;  // -20 deg
  const Eigen::Vector3f stray = 1.0f * Eigen::Vector3f(0.0f, std::cos(elevation[r]), std::sin(elevation[r]));
  xyz.insert(xyz.end(), {stray.x(), stray.y(), stray.z()});
  ring.push_back(r);

  ObstacleGrid og(scene_config(), {scene.lidar()});
  const SelfModel self = scene.self();
  og.add_frame(0, xyz.data(), ring.data(), ring.size(), 0.0, self);
  og.update(self);
  const Eigen::Vector3f at = stray + Eigen::Vector3f(0.0f, 0.0f, Scene::kHeight);
  const Eigen::Vector3f behind = 1.3f * stray + Eigen::Vector3f(0.0f, 0.0f, Scene::kHeight);
  EXPECT_NE(label_at(og, at.x(), at.y(), at.z()), kOccupied);
  EXPECT_EQ(label_at(og, behind.x(), behind.y(), behind.z()), kFree);

  // With the filter off, the same return is an obstacle hiding what's behind it.
  ObstacleGridParams keep_all;
  keep_all.min_cluster_size = 1;
  keep_all.min_cluster_rings = 1;
  ObstacleGrid unfiltered(scene_config(), {scene.lidar()}, keep_all);
  unfiltered.add_frame(0, xyz.data(), ring.data(), ring.size(), 0.0, self);
  unfiltered.update(self);
  EXPECT_EQ(label_at(unfiltered, at.x(), at.y(), at.z()), kOccupied);
  EXPECT_EQ(label_at(unfiltered, behind.x(), behind.y(), behind.z()), kUnknown);
}

TEST(ObstacleGrid, SmallClusterIsIgnored)
{
  Scene scene;
  std::vector<float> xyz;
  std::vector<uint16_t> ring;
  scene.cast(&xyz, &ring);
  // Three stray returns side by side (neighboring azimuth bins, 1 cm apart), 1 m out to the side.
  const std::vector<float> elevation = Scene::rings();
  const int r = 40;
  std::vector<Eigen::Vector3f> strays;
  for (int k = 0; k < 3; ++k) {
    const float azimuth = static_cast<float>(M_PI) / 2.0f + (k - 1) * kTwoPi / Scene::kAzimuthBins;
    strays.emplace_back(std::cos(elevation[r]) * std::cos(azimuth), std::cos(elevation[r]) * std::sin(azimuth),
      std::sin(elevation[r]));
  }
  for (const auto & p : strays) {
    xyz.insert(xyz.end(), {p.x(), p.y(), p.z()});
    ring.push_back(r);
  }
  const SelfModel self = scene.self();
  const Eigen::Vector3f at = strays[1] + Eigen::Vector3f(0.0f, 0.0f, Scene::kHeight);

  ObstacleGrid og(scene_config(), {scene.lidar()});
  og.add_frame(0, xyz.data(), ring.data(), ring.size(), 0.0, self);
  og.update(self);
  EXPECT_NE(label_at(og, at.x(), at.y(), at.z()), kOccupied);

  // Allowing clusters of 3 keeps them.
  ObstacleGridParams small;
  small.min_cluster_size = 3;
  small.min_cluster_rings = 1;  // the three are on one ring
  ObstacleGrid kept(scene_config(), {scene.lidar()}, small);
  kept.add_frame(0, xyz.data(), ring.data(), ring.size(), 0.0, self);
  kept.update(self);
  EXPECT_EQ(label_at(kept, at.x(), at.y(), at.z()), kOccupied);
}

TEST(ObstacleGrid, ReturnBehindRobotIsAGhost)
{
  Scene scene;
  std::vector<float> xyz;
  std::vector<uint16_t> ring;
  scene.cast(&xyz, &ring);
  // Push every return on the arm's beams 0.5 m farther out, behind the arm, as a multipath
  // reflection off the arm would.
  const SelfModel self = scene.self();
  std::vector<Eigen::Vector3f> ghosts;
  for (size_t i = 0; i < ring.size(); ++i) {
    Eigen::Vector3f p(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
    const Eigen::Vector3f in_base = p + Eigen::Vector3f(0.0f, 0.0f, Scene::kHeight);
    if (self.contains(in_base, 0.001f)) {
      p *= (p.norm() + 0.5f) / p.norm();
      xyz[3 * i] = p.x(); xyz[3 * i + 1] = p.y(); xyz[3 * i + 2] = p.z();
      ghosts.push_back(p + Eigen::Vector3f(0.0f, 0.0f, Scene::kHeight));
    }
  }
  ASSERT_GT(ghosts.size(), 20u);
  ObstacleGrid og(scene_config(), {scene.lidar()});
  og.add_frame(0, xyz.data(), ring.data(), ring.size(), 0.0, self);
  og.update(self);
  int occupied = 0;
  for (const auto & g : ghosts) {
    occupied += label_at(og, g.x(), g.y(), g.z()) == kOccupied;
  }
  EXPECT_EQ(occupied, 0);
  EXPECT_EQ(label_at(og, 0.8f, 0.0f, 0.8f), kShadow);  // still the arm's shadow
}

TEST(ObstacleGrid, VeilBeamsAreFreeUpToTheEdge)
{
  Scene scene;
  // A post 1 m out to the side, away from the arm. Next to its edge, the wall's returns behind
  // it look like veil points and are dropped, but their beams still crossed the space in front
  // of the post.
  scene.obstacles.emplace_back(Eigen::Vector3f(-0.1f, 1.0f, 0.0f), Eigen::Vector3f(0.1f, 1.1f, 1.0f));
  const auto labeled = label_scene(scene);
  const ObstacleGrid & og = *labeled;
  // Azimuth 84.1 deg is just past the edge (84.3 deg); ring 40 points 20 deg down.
  const float azimuth = 84.1f * kPi / 180.0f, elevation = -20.0f * kPi / 180.0f;
  const RangeImage & image = og.range_image(0);
  const size_t cell = 40 * image.cols + static_cast<int>((azimuth + kPi) / kTwoPi * image.cols);
  EXPECT_EQ(image.hit[cell], Hit::kPassed);
  EXPECT_NEAR(image.range[cell], 1.0f / std::cos(elevation), 0.05f);  // the post's near face
  const Eigen::Vector3f front(0.6f * std::cos(elevation) * std::cos(azimuth),
    0.6f * std::cos(elevation) * std::sin(azimuth), Scene::kHeight + 0.6f * std::sin(elevation));
  EXPECT_EQ(label_at(og, front.x(), front.y(), front.z()), kFree);

  // Without the veil filter the wall's return stays.
  ObstacleGridParams no_veil;
  no_veil.veil_angle = 0.0f;
  EXPECT_EQ(label_scene(scene, no_veil)->range_image(0).hit[cell], Hit::kOther);
}

// Frames of a scene with a post that stays put in the world while the base turns `yaw_rate`
// (rad/s), one frame every 0.1 s from t = 0 to `duration`. Returns the grid after the last.
std::unique_ptr<ObstacleGrid> watch_post(float yaw_rate, float duration, float post_appears = 0.0f)
{
  auto og = std::make_unique<ObstacleGrid>(scene_config(), std::vector<LidarModel>{Scene().lidar()});
  for (float t = 0.0f; t <= duration + 1e-4f; t += 0.1f) {
    const float yaw = yaw_rate * t;
    Scene scene;
    if (t >= post_appears) {
      // At world (0, 1), so at angle 90 deg - yaw from the base.
      const Eigen::Vector2f at = Eigen::Rotation2Df(-yaw) * Eigen::Vector2f(0.0f, 1.0f);
      scene.obstacles.emplace_back(Eigen::Vector3f(at.x() - 0.05f, at.y() - 0.05f, 0.0f),
        Eigen::Vector3f(at.x() + 0.05f, at.y() + 0.05f, 1.0f));
    }
    std::vector<float> xyz;
    std::vector<uint16_t> ring;
    scene.cast(&xyz, &ring);
    const SelfModel self = scene.self();
    og->add_frame(0, xyz.data(), ring.data(), ring.size(), t, self);
    Eigen::Isometry3f world_from_base = Eigen::Isometry3f::Identity();
    world_from_base.rotate(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
    og->update(self, world_from_base);
  }
  return og;
}

// How many voxels have each label.
std::array<int, 7> label_counts(const ObstacleGrid & og)
{
  std::array<int, 7> counts{};
  for (uint8_t label : og.labels()) {++counts[label];}
  return counts;
}

TEST(ObstacleGrid, ObstaclesThatStayPutBecomeStatic)
{
  // Before static_after (3 s) the post may move; after it, it's static.
  const auto early = label_counts(*watch_post(0.0f, 2.5f));
  EXPECT_GT(early[kOccupied], 50);
  EXPECT_EQ(early[kStatic], 0);
  const auto late = label_counts(*watch_post(0.0f, 3.5f));
  EXPECT_EQ(late[kOccupied], 0);
  EXPECT_GT(late[kStatic], 50);
}

TEST(ObstaclesThatStayPutBecomeStatic, EvenWhileTheBaseTurns)
{
  // 0.3 rad/s for 4 s: the post sweeps 69 deg around the grid, but stays put in the world.
  const auto counts = label_counts(*watch_post(0.3f, 4.0f));
  EXPECT_GT(counts[kStatic], 4 * counts[kOccupied]);
}

TEST(ObstaclesThatStayPutBecomeStatic, NewObstaclesAreNot)
{
  const auto counts = label_counts(*watch_post(0.0f, 4.0f, 2.0f));  // appears at 2 s
  EXPECT_GT(counts[kOccupied], 50);
  EXPECT_EQ(counts[kStatic], 0);
}

// --- Motion checks ---------------------------------------------------------------------------

SelfShape box_shape(const Eigen::Vector3f & center, const Eigen::Vector3f & half, float yaw = 0.0f)
{
  Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
  pose.rotate(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
  pose.translate(center);  // after the rotation: the box turns about the axis with it
  return SelfShape{SelfShape::Type::kBox, pose, half};
}

// True if p is inside the box, angles compared modulo a turn.
bool inside(const CylinderBox & box, const Eigen::Vector3f & p, float tolerance = 1e-5f)
{
  const float r = p.head<2>().norm();
  if (r < box.r_min - tolerance || r > box.r_max + tolerance || p.z() < box.z_min - tolerance ||
    p.z() > box.z_max + tolerance)
  {
    return false;
  }
  if (box.theta_max - box.theta_min >= kTwoPi) {return true;}
  float offset = std::fmod(std::atan2(p.y(), p.x()) - box.theta_min, kTwoPi);
  if (offset < 0.0f) {offset += kTwoPi;}
  return offset <= box.theta_max - box.theta_min + tolerance || offset >= kTwoPi - tolerance;
}

TEST(MotionCheck, CylinderBoundsOfBoxBesideAxis)
{
  const SelfShape box = box_shape(Eigen::Vector3f(1.0f, 0.0f, 0.5f), Eigen::Vector3f(0.1f, 0.1f, 0.1f));
  const CylinderBox b = cylinder_bounds(piece_corners(box));
  EXPECT_NEAR(b.theta_min, -std::atan2(0.1f, 0.9f), 1e-5f);
  EXPECT_NEAR(b.theta_max, std::atan2(0.1f, 0.9f), 1e-5f);
  EXPECT_NEAR(b.r_min, 0.9f, 1e-5f);  // the near face, not a corner
  EXPECT_NEAR(b.r_max, std::hypot(1.1f, 0.1f), 1e-5f);
  EXPECT_NEAR(b.z_min, 0.4f, 1e-5f);
  EXPECT_NEAR(b.z_max, 0.6f, 1e-5f);

  // Its far half alone.
  const CylinderBox far = cylinder_bounds(piece_corners(box, Eigen::Vector3f(0.5f, 0.0f, 0.0f)));
  EXPECT_NEAR(far.r_min, 1.0f, 1e-5f);
  EXPECT_NEAR(far.theta_max, std::atan2(0.1f, 1.0f), 1e-5f);
}

TEST(MotionCheck, CylinderBoundsAroundAxisAreAFullTurn)
{
  const SelfShape mast = box_shape(Eigen::Vector3f(0.01f, 0.0f, 1.0f), Eigen::Vector3f(0.04f, 0.02f, 0.7f));
  const CylinderBox b = cylinder_bounds(piece_corners(mast));
  EXPECT_GE(b.theta_max - b.theta_min, kTwoPi);
  EXPECT_EQ(b.r_min, 0.0f);
}

TEST(MotionCheck, CylinderBoundsAcrossPiStayNarrow)
{
  const SelfShape box = box_shape(Eigen::Vector3f(-1.0f, 0.0f, 0.5f), Eigen::Vector3f(0.1f, 0.1f, 0.1f));
  const CylinderBox b = cylinder_bounds(piece_corners(box));
  EXPECT_NEAR(b.theta_max - b.theta_min, 2.0f * std::atan2(0.1f, 0.9f), 1e-4f);
  EXPECT_TRUE(inside(b, Eigen::Vector3f(-1.0f, 0.0f, 0.5f)));
}

TEST(MotionCheck, GrownBoxHoldsEverythingWithinDistance)
{
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> unit(0.0f, 1.0f);
  for (float d : {0.02f, 0.1f, 0.3f}) {
    for (const CylinderBox box : {CylinderBox{-0.2f, 0.3f, 0.5f, 0.7f, 0.2f, 0.4f},
        CylinderBox{3.0f, 3.4f, 0.15f, 0.3f, 1.0f, 1.1f}})
    {
      const CylinderBox grown = grow(box, d);
      for (int i = 0; i < 2000; ++i) {
        const float theta = box.theta_min + unit(rng) * (box.theta_max - box.theta_min);
        const float r = box.r_min + unit(rng) * (box.r_max - box.r_min);
        const Eigen::Vector3f q(r * std::cos(theta), r * std::sin(theta),
          box.z_min + unit(rng) * (box.z_max - box.z_min));
        const Eigen::Vector3f offset = Eigen::Vector3f::Random().normalized() * d * unit(rng);
        ASSERT_TRUE(inside(grown, q + offset)) << "d " << d << ", point " << (q + offset).transpose();
      }
    }
  }
}

TEST(MotionCheck, MergeTakesTheShortWayAround)
{
  const CylinderBox a{6.2f, 6.25f, 1.0f, 1.1f, 0.0f, 0.1f};
  const CylinderBox b{0.05f, 0.1f, 0.9f, 1.0f, 0.2f, 0.3f};
  const CylinderBox m = merge(a, b);
  EXPECT_NEAR(m.theta_max - m.theta_min, 0.1f + kTwoPi - 6.2f, 1e-5f);
  EXPECT_FLOAT_EQ(m.r_min, 0.9f);
  EXPECT_FLOAT_EQ(m.z_max, 0.3f);
}

TEST(MotionCheck, AllowedSpeedSolvesTheStoppingBudget)
{
  const MotionLimits limits;
  EXPECT_FLOAT_EQ(allowed_speed(0.0f, limits.contact_speed, limits.obstacle_speed, limits.deceleration, limits), limits.contact_speed);
  EXPECT_FLOAT_EQ(allowed_speed(-1.0f, 0.0f, limits.obstacle_speed, limits.deceleration, limits), 0.0f);
  EXPECT_FLOAT_EQ(allowed_speed(100.0f, 0.0f, limits.obstacle_speed, limits.deceleration, limits), limits.max_speed);

  const float clearance = 1.2f, vf = limits.contact_speed;
  const float v = allowed_speed(clearance, vf, limits.obstacle_speed, limits.deceleration, limits);
  ASSERT_GT(v, vf);
  ASSERT_LT(v, limits.max_speed);
  const float a = limits.deceleration, t = limits.latency, vo = limits.obstacle_speed;
  const float needed = (vo + v) * t + vo * (v - vf) / a + (v * v - vf * vf) / (2.0f * a);
  EXPECT_NEAR(needed, clearance, 1e-4f);
  EXPECT_GT(allowed_speed(clearance + 0.1f, vf, limits.obstacle_speed, limits.deceleration, limits), v);
}

// The scene's arm, as a body sliding along +x by `extension`.
SelfModel arm_at(const Scene & scene, float extension)
{
  return SelfModel({box_shape(scene.arm.center() + Eigen::Vector3f(extension, 0.0f, 0.0f), scene.arm.sizes() / 2.0f)});
}

TEST(MotionCheck, SlowsApproachingAnObstacleAndIsBlockedByIt)
{
  Scene scene;
  // A post in front of the arm at x = 1.0..1.1, as tall as the arm is high.
  scene.obstacles.emplace_back(Eigen::Vector3f(1.0f, -0.1f, 0.0f), Eigen::Vector3f(1.1f, 0.1f, 1.0f));
  const auto labeled = label_scene(scene);

  // The arm's front face (x = 0.5) extends to 1.05, 1 cm at a time.
  std::vector<SelfModel> samples;
  for (int i = 0; i <= 55; ++i) {samples.push_back(arm_at(scene, 0.01f * i));}
  MotionLimits limits;
  limits.max_clearance = 0.6f;
  limits.obstacle_speed = 0.3f;  // slow enough that the speeds vary over this short motion
  limits.unsafe_obstacle_speed = 0.3f;
  const std::vector<StepLimit> steps = MotionCheck(limits).check(*labeled, samples);
  ASSERT_EQ(steps.size(), 55u);

  for (size_t k = 0; k < steps.size(); ++k) {
    EXPECT_NEAR(steps[k].travel, 0.01f, 1e-5f);
    EXPECT_EQ(steps[k].shape, 0);
    EXPECT_NEAR(steps[k].min_duration, steps[k].travel / steps[k].speed, 1e-5f);
    if (k > 0) {
      // Either of two bounds may be the tighter one from step to step, so allow a little.
      EXPECT_LE(steps[k].clearance_occupied, steps[k - 1].clearance_occupied + 0.015f);
    }
  }
  // At the start the front face is 0.5 m from the post (less voxel rounding and half a step).
  EXPECT_NEAR(steps.front().clearance_occupied, 0.5f, 0.04f);
  EXPECT_FALSE(steps.front().blocked);
  // The LiDAR's near field counts as the robot's own shadow, not unknown.
  EXPECT_GT(steps.front().clearance_unsafe, 0.15f);
  EXPECT_FALSE(steps.front().unsafe);
  EXPECT_GT(steps.front().speed, limits.unsafe_speed);
  // Reaching it, it's blocked, at no more than contact speed.
  EXPECT_TRUE(steps.back().blocked);
  EXPECT_LE(steps.back().speed, limits.contact_speed);
}

TEST(MotionCheck, ClearanceNearTheAxisDoesNotWrapAround)
{
  Scene scene;
  // A post on the far side of the axis from a small moving box, both 0.5 m out (beside the
  // arm), 0.9 m apart. Grown in cylindrical coordinates, the box would span a full turn and
  // reach the post after about 0.23 m. The x, y, z tables round out by a cell and a voxel.
  scene.obstacles.emplace_back(Eigen::Vector3f(-0.05f, -0.55f, 0.0f), Eigen::Vector3f(0.05f, -0.45f, 1.0f));
  const auto labeled = label_scene(scene);
  const Eigen::Vector3f half(0.05f, 0.05f, 0.05f);
  MotionLimits limits;
  limits.max_clearance = 1.2f;
  const std::vector<StepLimit> steps = MotionCheck(limits).check(*labeled, {
      SelfModel({box_shape(Eigen::Vector3f(0.0f, 0.50f, 0.5f), half)}),
      SelfModel({box_shape(Eigen::Vector3f(0.0f, 0.51f, 0.5f), half)})});
  ASSERT_EQ(steps.size(), 1u);
  EXPECT_GT(steps[0].clearance_occupied, 0.8f);
  EXPECT_LT(steps[0].clearance_occupied, 0.91f);
}

TEST(MotionCheck, FasterBrakingAllowsMoreSpeed)
{
  Scene scene;
  scene.obstacles.emplace_back(Eigen::Vector3f(1.0f, -0.1f, 0.0f), Eigen::Vector3f(1.1f, 0.1f, 1.0f));
  const auto labeled = label_scene(scene);
  MotionLimits limits;
  limits.obstacle_speed = 0.0f;  // only braking counts
  const std::vector<SelfModel> samples = {arm_at(scene, 0.0f), arm_at(scene, 0.01f)};
  // The arm moves 1 cm; a joint moving 1 cm with deceleration a gives braking 0.01 / a.
  const StepLimit slow = MotionCheck(limits).check(*labeled, samples, 0.0f, 0.0f, {0.01f / 0.3f})[0];
  const StepLimit fast = MotionCheck(limits).check(*labeled, samples, 0.0f, 0.0f, {0.01f / 3.0f})[0];
  const StepLimit fallback = MotionCheck(limits).check(*labeled, samples)[0];
  // Whichever of the obstacles and unsafe space (here the LiDAR's blind spots past the post)
  // allows less.
  auto expected = [&](const StepLimit & step, float decel) {
      return std::min(allowed_speed(step.clearance_occupied, limits.contact_speed, 0.0f, decel, limits),
               allowed_speed(step.clearance_unsafe, limits.unsafe_speed, 0.0f, decel, limits));
    };
  EXPECT_NEAR(slow.speed, expected(slow, 0.3f), 1e-5f);
  EXPECT_NEAR(fast.speed, expected(fast, 3.0f), 1e-5f);
  EXPECT_GT(fast.speed, slow.speed);
  EXPECT_NEAR(fallback.speed, expected(fallback, limits.deceleration), 1e-5f);
}

TEST(MotionCheck, StandoffStopsShortOfObstacles)
{
  Scene scene;
  scene.obstacles.emplace_back(Eigen::Vector3f(1.0f, -0.1f, 0.0f), Eigen::Vector3f(1.1f, 0.1f, 1.0f));
  const auto labeled = label_scene(scene);
  MotionLimits limits;
  limits.standoff = 0.1f;
  // Front face from 0.85 to 0.86: 0.14 m short of the post, inside the standoff plus the
  // distance an obstacle covers during the latency.
  const std::vector<StepLimit> steps = MotionCheck(limits).check(*labeled, {arm_at(scene, 0.35f), arm_at(scene, 0.36f)});
  ASSERT_EQ(steps.size(), 1u);
  EXPECT_FALSE(steps[0].blocked);
  EXPECT_EQ(steps[0].speed, 0.0f);
  EXPECT_TRUE(std::isinf(steps[0].min_duration));
}

TEST(MotionCheck, StillShapesAreNotChecked)
{
  Scene scene;
  scene.obstacles.emplace_back(Eigen::Vector3f(0.5f, -0.1f, 0.0f), Eigen::Vector3f(0.6f, 0.1f, 1.0f));
  const auto labeled = label_scene(scene);
  // The arm touches the post but doesn't move.
  const std::vector<StepLimit> steps = MotionCheck().check(*labeled, {arm_at(scene, 0.0f), arm_at(scene, 0.0f)});
  ASSERT_EQ(steps.size(), 1u);
  EXPECT_EQ(steps[0].shape, -1);
  EXPECT_EQ(steps[0].min_duration, 0.0f);
  EXPECT_FALSE(steps[0].blocked);
}

TEST(MotionCheck, BaseRotationSweepsTheAnglesBetween)
{
  Scene scene;
  // A post at 90 degrees, 1 m out; the arm's box out at 0.7-0.9 m turns from 0 to 180 degrees.
  scene.obstacles.emplace_back(Eigen::Vector3f(-0.05f, 0.95f, 0.0f), Eigen::Vector3f(0.05f, 1.05f, 1.0f));
  const auto labeled = label_scene(scene);
  const Eigen::Vector3f center(0.8f, 0.0f, 0.9f), half(0.1f, 0.05f, 0.05f);
  // Two samples half a turn apart would be ambiguous, so take three.
  std::vector<SelfModel> samples;
  for (float yaw : {0.0f, 1.4f, 2.8f}) {samples.push_back(SelfModel({box_shape(center, half, yaw)}));}
  const std::vector<StepLimit> steps = MotionCheck().check(*labeled, samples);
  ASSERT_EQ(steps.size(), 2u);
  EXPECT_TRUE(steps[0].blocked);  // the post sits at 90 degrees, inside the first step's sweep
  EXPECT_TRUE(steps[1].blocked);
  EXPECT_GT(steps[0].travel, 0.9f * 2.0f * std::sin(0.7f));
}

