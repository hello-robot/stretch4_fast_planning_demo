// The obstacle grid: LiDAR frames in, one label per voxel and box-count tables out.
//
// Labels come from the latest frame of every LiDAR:
//   kOccupied            a return from something that isn't the robot or the floor, which
//                        no newer frame from another LiDAR sees through, and which may move
//   kStatic              occupied, and has been where it is (in the world, not just relative
//                        to the turning base) for at least static_after
//   kSelf                the robot's own body: inside its shapes, or holding a return off it
//   kFree                some LiDAR's rays went past it
//   kShadow              a blind spot the robot makes: hidden only by the robot (mostly under
//                        the arm), or too near a LiDAR to see; probably free
//   kContaminatedShadow  shadow within contamination_reach of an obstacle (and of unknown
//                        space, if unknown_contaminates_shadow), which a person could be
//                        reaching in from, so it counts as possibly occupied
//   kUnknown             hidden by something else, or not seen at all
// "Obstacles" means occupied or static. "Unsafe" means unknown or contaminated shadow: the
// robot moves through it only at contact speed.

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "stretch_obstacle_grid/cartesian_table.hpp"
#include "stretch_obstacle_grid/cylindrical_grid.hpp"
#include "stretch_obstacle_grid/lidar.hpp"
#include "stretch_obstacle_grid/self_model.hpp"
#include "stretch_obstacle_grid/volume_table.hpp"

namespace stretch_obstacle_grid
{

enum Label : uint8_t
{
  kFree = 0,
  kOccupied = 1,
  kSelf = 2,
  kShadow = 3,
  kContaminatedShadow = 4,
  kUnknown = 5,
  kStatic = 6,
};

struct ObstacleGridParams
{
  float self_point_padding = 0.03f;  // m: returns this close to the robot's shapes are the robot
  float self_voxel_padding = 0.01f;  // m: voxels whose centers are this close are the robot
  float range_margin = 0.02f;        // m: range noise allowance when comparing rays to voxels
  int dropout_min_neighbors = 4;     // see RangeImage::fill_dropouts; 9 turns filling off
  // The robot's dark, glossy covers often return nothing. An empty ray whose path hits the
  // robot's shapes counts as hitting the robot there, so it hides what's behind as shadow
  // rather than leaving it unknown. So does a veil point's beam that reaches the robot.
  bool empty_rays_hit_self = true;
  // How far (m) contamination reaches into a shadow from where the shadow touches unknown
  // space or an obstacle: a hidden person has to connect to something visible within about
  // a body part's length. Infinity contaminates the whole connected shadow.
  float contamination_reach = 0.20f;
  // Whether shadow touching unknown space is contaminated too (from there, not just from
  // obstacles). Off: a person can't hide entirely in the robot's blind spots, and if they're
  // partly in view, the obstacle they make contaminates it.
  bool unknown_contaminates_shadow = false;
  // Voxels this close (m) to a LiDAR that no LiDAR sees count as the robot's shadow: nothing
  // returns from within min_return_range, so the head's surroundings are always blind.
  float self_blind_range = 0.40f;
  // Obstacles that have stayed put this long (s) are static: the robot can assume they won't
  // come at it. Tracked in cells of static_cell (m) in the world frame passed to update(), so
  // turning the base doesn't make them new; a cell may go unseen for up to static_gap (s).
  // Each obstacle voxel marks the cells within half a cell of its center, so anything within
  // about a cell of a static obstacle's surface may count as static too.
  float static_after = 3.0f;
  float static_cell = 0.05f;
  float static_gap = 0.5f;
  int table_threads = 8;  // OpenMP threads for building the box-count tables
  int threads = 0;        // OpenMP threads for the per-return and per-voxel loops (0: default)
  // In dual-return mode the Hesai repeats each firing's points, identical 99.9% of the time.
  // A point bitwise identical to the one this many points earlier is skipped; set it to the
  // ring count for dual-return frames, 0 to keep every point.
  int duplicate_stride = 0;
  // Small clusters of returns (reflections and speckle, mostly around the gripper) count as
  // no return. Returns in neighboring range image cells belong to the same cluster when
  // they're within cluster_radius + cluster_radius_per_m * range of each other (neighboring
  // beams spread apart with range, about 2 cm at 1.5 m); clusters of fewer than
  // min_cluster_size returns are dropped. 1 keeps everything.
  float cluster_radius = 0.03f;        // m
  float cluster_radius_per_m = 0.02f;  // m per m of range
  int min_cluster_size = 5;
  // Clusters must also span this many rings: some single rings return lines of points well
  // off the surface the rings around them see (a channel misreading). Anything the size of
  // a hand spans several rings (0.7 deg apart); a thin horizontal cord might not.
  int min_cluster_rings = 2;
  // Veil points ("mixed pixels"): a beam that clips an edge returns a range between the edge
  // and what's behind it, drawing a dim streak of fake points behind every edge. A return is
  // dropped when a nearer neighboring return lies within veil_angle of straight along its
  // beam, i.e. the two are stacked in depth rather than side by side on a surface. Real edges
  // (the nearer returns) are kept, and ghosts behind the robot don't count as edges. The
  // dropped return's beam still got as far as the edge, so the space in front of the edge stays
  // free and only the space behind it is unknown. 0 turns this off.
  float veil_angle = 0.1745f;  // rad (10 deg)
  // Returns nearer than this (m) count as no return: the Hesai's near field is unreliable,
  // and only the robot's own head and mast are this close to the LiDARs.
  float min_return_range = 0.35f;
};

// Milliseconds spent in each step of the last add_frame and update.
struct Timing
{
  double bin_points = 0.0;  // all of add_frame; the steps below add up to it
  double nearest = 0.0;     // nearest return per range image cell
  double clusters = 0.0;    // dropping small clusters
  double classify = 0.0;    // what each return hit
  double veil = 0.0;        // veil points, and binning occupancy
  double empty_rays = 0.0;  // empty rays toward the robot, and filling dropouts
  double self_voxels = 0.0;
  double label = 0.0;
  double contaminate = 0.0;
  double tables = 0.0;
  double static_obstacles = 0.0;
};

class ObstacleGrid
{
public:
  ObstacleGrid(
    const GridConfig & config, const std::vector<LidarModel> & lidars,
    const ObstacleGridParams & params = ObstacleGridParams());
  // The tables refer to grid_, so the grid can't be copied or moved.
  ObstacleGrid(const ObstacleGrid &) = delete;
  ObstacleGrid & operator=(const ObstacleGrid &) = delete;

  const CylindricalGrid & grid() const {return grid_;}
  int lidar_count() const {return static_cast<int>(lidars_.size());}
  static constexpr int kMaxLidars = 8;  // occupancy is tracked per LiDAR in a byte

  // The parts of the robot that never move relative to the base (base, mast, head), set once.
  // They're rasterized here, so the SelfModel passed to add_frame and update only needs the
  // moving parts. Both count as the robot everywhere.
  void set_static_self(const SelfModel & self);

  // Take in one frame from LiDAR `lidar`: n points in that LiDAR's frame (non-finite where
  // there was no return), their ring indices, the frame's start time (s), and the robot's
  // body during the frame. Replaces that LiDAR's previous frame.
  void add_frame(
    int lidar, const float * xyz, const uint16_t * ring, size_t n, double stamp, const SelfModel & self);

  // Relabel every voxel from the latest frames, with the robot's body as it is now, and
  // rebuild the tables. world_from_base: base_footprint in a frame fixed in the world (odom),
  // for telling static obstacles from moving ones.
  void update(const SelfModel & self, const Eigen::Isometry3f & world_from_base = Eigen::Isometry3f::Identity());

  const std::vector<uint8_t> & labels() const {return labels_;}
  // Box counts of obstacles (occupied or static), of the ones that may move (occupied), and
  // of unsafe space, for boxes in (theta, r, z) and in x, y, z.
  uint32_t count_occupied(const CylinderBox & box) const {return occupied_table_.count(box);}
  uint32_t count_unsafe(const CylinderBox & box) const {return unsafe_table_.count(box);}
  const VolumeTable & occupied_table() const {return occupied_table_;}
  const VolumeTable & moving_table() const {return moving_table_;}
  const VolumeTable & unsafe_table() const {return unsafe_table_;}
  const CartesianTable & occupied_box_table() const {return occupied_box_table_;}
  const CartesianTable & moving_box_table() const {return moving_box_table_;}
  const CartesianTable & unsafe_box_table() const {return unsafe_box_table_;}

  // Start time of the oldest frame behind the current labels: obstacles may have moved
  // since then, which the caller pads for.
  double oldest_stamp() const;

  const RangeImage & range_image(int lidar) const {return frames_[lidar].image;}
  RayView ray_view(int lidar, size_t voxel) const;
  const Timing & timing() const {return timing_;}

private:
  struct Frame
  {
    RangeImage image;
    std::vector<size_t> occupied;  // voxels holding a return from something other than the robot
    double stamp = -1.0;
  };

  Label label_voxel(size_t voxel) const;
  void mark_static_obstacles(const Eigen::Isometry3f & world_from_base);
  void contaminate_shadows();
  bool is_self(const SelfModel & moving, const Eigen::Vector3f & p) const;
  void drop_small_clusters(
    const float * xyz, const Eigen::Isometry3f & base_from_lidar, const SelfModel & self, int rows, int cols);
  // Turns veil points' cells in `image` into kPassed at the edge in front of them.
  void drop_veil_points(const float * xyz, int rows, int cols, RangeImage * image);

  CylindricalGrid grid_;
  ObstacleGridParams params_;
  std::vector<LidarModel> lidars_;
  std::vector<std::unique_ptr<RayTable>> rays_;
  std::vector<Frame> frames_;

  SelfModel static_self_;
  std::vector<uint8_t> static_self_flags_;            // static_self_ rasterized
  std::vector<uint8_t> self_flags_;    // scratch for update(): the robot's body now
  std::vector<uint8_t> occupied_by_;   // scratch for update(): bit i set if LiDAR i's frame has a return here
  // Scratch for add_frame: each range image cell's nearest return (point index, or -1), and
  // the clusters they form (union-find parents and sizes, per cell).
  std::vector<int32_t> nearest_in_cell_;
  std::vector<uint32_t> cluster_parent_, cluster_size_;
  std::vector<uint16_t> cluster_row_min_, cluster_row_max_;
  std::vector<uint8_t> unclustered_;  // per cell: its return is on the floor or the robot
  std::vector<float> veil_range_;  // per cell: the edge in front of its veil point (m), or NaN
  std::vector<uint8_t> veil_off_robot_;  // per cell: that edge is the robot
  std::vector<uint8_t> labels_;
  VolumeTable occupied_table_, moving_table_, unsafe_table_;
  CartesianTable occupied_box_table_, moving_box_table_, unsafe_box_table_;
  std::vector<uint8_t> near_lidar_;  // per voxel: within self_blind_range of a LiDAR
  // World cells where obstacles have been: (first, last) time seen, with no gap over static_gap.
  std::unordered_map<int64_t, std::pair<double, double>> obstacle_cells_;
  Timing timing_;
};

}  // namespace stretch_obstacle_grid
