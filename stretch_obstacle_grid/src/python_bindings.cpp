// Python bindings: `import stretch_obstacle_grid`.

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "stretch_obstacle_grid/motion_check.hpp"
#include "stretch_obstacle_grid/obstacle_grid.hpp"

namespace py = pybind11;
using namespace stretch_obstacle_grid;

namespace
{

Eigen::Isometry3f to_isometry(const Eigen::Matrix4f & matrix)
{
  Eigen::Isometry3f pose;
  pose.matrix() = matrix;
  return pose;
}

SelfShape make_shape(SelfShape::Type type, const Eigen::Matrix4f & pose, Eigen::Vector3f dims)
{
  return SelfShape{type, to_isometry(pose), dims};
}

// Voxel centers (x, y, z) in base_footprint, one row per voxel.
py::array_t<float> voxel_centers(const CylindricalGrid & grid)
{
  py::array_t<float> out({static_cast<py::ssize_t>(grid.size()), py::ssize_t{3}});
  auto view = out.mutable_unchecked<2>();
  for (size_t i = 0; i < grid.size(); ++i) {
    grid.center(i, &view(i, 0), &view(i, 1), &view(i, 2));
  }
  return out;
}

// Voxel cells (radial bin, theta bin within its band, z bin), one row per voxel.
py::array_t<int32_t> voxel_cells(const CylindricalGrid & grid)
{
  py::array_t<int32_t> out({static_cast<py::ssize_t>(grid.size()), py::ssize_t{3}});
  auto view = out.mutable_unchecked<2>();
  for (size_t i = 0; i < grid.size(); ++i) {
    const Cell c = grid.cell(i);
    view(i, 0) = c.ri;
    view(i, 1) = c.ti;
    view(i, 2) = c.zi;
  }
  return out;
}

}  // namespace

PYBIND11_MODULE(stretch_obstacle_grid, m)
{
  m.doc() = "Cylindrical obstacle grid around Stretch 4's base axis, built from LiDAR frames.";

  py::class_<GridConfig>(m, "GridConfig")
  .def(py::init<>())
  .def_readwrite("r_min", &GridConfig::r_min)
  .def_readwrite("r_max", &GridConfig::r_max)
  .def_readwrite("z_min", &GridConfig::z_min)
  .def_readwrite("z_max", &GridConfig::z_max)
  .def_readwrite("cell", &GridConfig::cell);

  py::class_<Band>(m, "Band")
  .def_readonly("theta_bins", &Band::theta_bins)
  .def_readonly("dtheta", &Band::dtheta)
  .def_readonly("r_begin", &Band::r_begin)
  .def_readonly("r_end", &Band::r_end)
  .def_readonly("offset", &Band::offset)
  .def("__repr__", [](const Band & b) {
      return "Band(theta_bins=" + std::to_string(b.theta_bins) + ", radial bins " +
      std::to_string(b.r_begin) + "-" + std::to_string(b.r_end) + ")";
    });

  py::class_<CylindricalGrid>(m, "CylindricalGrid")
  .def(py::init<const GridConfig &>(), py::arg("config") = GridConfig())
  .def_property_readonly("config", &CylindricalGrid::config)
  .def_property_readonly("size", &CylindricalGrid::size)
  .def_property_readonly("radial_bins", &CylindricalGrid::radial_bins)
  .def_property_readonly("z_bins", &CylindricalGrid::z_bins)
  .def_property_readonly("bands", &CylindricalGrid::bands)
  .def("index", &CylindricalGrid::index, py::arg("ri"), py::arg("ti"), py::arg("zi"))
  .def("locate", [](const CylindricalGrid & g, float x, float y, float z) -> py::object {
      size_t index;
      return g.locate(x, y, z, &index) ? py::cast(index) : py::none();
    }, py::arg("x"), py::arg("y"), py::arg("z"), "Voxel index containing (x, y, z), or None.")
  .def("centers", &voxel_centers, "Voxel centers in base_footprint, shape (size, 3).")
  .def("cells", &voxel_cells, "Voxel (radial, theta, z) bins, shape (size, 3).");

  py::class_<CylinderBox>(m, "CylinderBox")
  .def(py::init<float, float, float, float, float, float>(),
    py::arg("theta_min"), py::arg("theta_max"), py::arg("r_min"), py::arg("r_max"),
    py::arg("z_min"), py::arg("z_max"))
  .def_readwrite("theta_min", &CylinderBox::theta_min)
  .def_readwrite("theta_max", &CylinderBox::theta_max)
  .def_readwrite("r_min", &CylinderBox::r_min)
  .def_readwrite("r_max", &CylinderBox::r_max)
  .def_readwrite("z_min", &CylinderBox::z_min)
  .def_readwrite("z_max", &CylinderBox::z_max)
  .def("__repr__", [](const CylinderBox & b) {
      return "CylinderBox(theta " + std::to_string(b.theta_min) + ".." + std::to_string(b.theta_max) +
      ", r " + std::to_string(b.r_min) + ".." + std::to_string(b.r_max) +
      ", z " + std::to_string(b.z_min) + ".." + std::to_string(b.z_max) + ")";
    });

  py::class_<LidarModel>(m, "LidarModel")
  .def(py::init([](const Eigen::Matrix4f & base_from_lidar, std::vector<float> ring_elevation, int azimuth_bins) {
      return LidarModel{to_isometry(base_from_lidar), std::move(ring_elevation), azimuth_bins};
    }), py::arg("base_from_lidar"), py::arg("ring_elevation"), py::arg("azimuth_bins") = 720)
  .def_readonly("ring_elevation", &LidarModel::ring_elevation)
  .def_readonly("azimuth_bins", &LidarModel::azimuth_bins);

  py::class_<SelfShape>(m, "SelfShape")
  .def_property_readonly("kind", [](const SelfShape & s) {
      switch (s.type) {
        case SelfShape::Type::kBox: return "box";
        case SelfShape::Type::kCylinder: return "cylinder";
        case SelfShape::Type::kSphere: return "sphere";
        case SelfShape::Type::kCapsule: return "capsule";
      }
      return "?";
    })
  .def_property_readonly("pose", [](const SelfShape & s) {return Eigen::Matrix4f(s.pose.matrix());},
    "4x4 pose of the shape in base_footprint.")
  .def_property_readonly("dims", [](const SelfShape & s) {return Eigen::Vector3f(s.dims);},
    "Box: half sizes. Cylinder and capsule: (radius, half length, -). Sphere: (radius, -, -).")
  .def_static("box", [](const Eigen::Matrix4f & pose, const Eigen::Vector3f & half_sizes) {
      return make_shape(SelfShape::Type::kBox, pose, half_sizes);
    }, py::arg("pose"), py::arg("half_sizes"))
  .def_static("cylinder", [](const Eigen::Matrix4f & pose, float radius, float half_length) {
      return make_shape(SelfShape::Type::kCylinder, pose, Eigen::Vector3f(radius, half_length, 0.0f));
    }, py::arg("pose"), py::arg("radius"), py::arg("half_length"))
  .def_static("capsule", [](const Eigen::Matrix4f & pose, float radius, float half_length) {
      return make_shape(SelfShape::Type::kCapsule, pose, Eigen::Vector3f(radius, half_length, 0.0f));
    }, py::arg("pose"), py::arg("radius"), py::arg("half_length"))
  .def_static("sphere", [](const Eigen::Matrix4f & pose, float radius) {
      return make_shape(SelfShape::Type::kSphere, pose, Eigen::Vector3f(radius, 0.0f, 0.0f));
    }, py::arg("pose"), py::arg("radius"));

  py::class_<SelfModel>(m, "SelfModel")
  .def(py::init<>())
  .def(py::init<std::vector<SelfShape>>(), py::arg("shapes"))
  .def("contains", [](const SelfModel & s, const Eigen::Vector3f & p, float padding) {
      return s.contains(p, padding);
    }, py::arg("point"), py::arg("padding") = 0.0f)
  .def("__len__", [](const SelfModel & s) {return s.shapes().size();})
  .def("shapes", &SelfModel::shapes, "A copy of the shapes.")
  .def("ray_distance", &SelfModel::ray_distance, py::arg("origin"), py::arg("direction"), py::arg("padding") = 0.0f,
    "Distance along a ray (unit direction) to the first shape (grown by padding) it enters from outside, or inf.");

  py::class_<ObstacleGridParams>(m, "ObstacleGridParams")
  .def(py::init<>())
  .def_readwrite("self_point_padding", &ObstacleGridParams::self_point_padding)
  .def_readwrite("self_voxel_padding", &ObstacleGridParams::self_voxel_padding)
  .def_readwrite("range_margin", &ObstacleGridParams::range_margin)
  .def_readwrite("graze_margin", &ObstacleGridParams::graze_margin)
  .def_readwrite("dropout_min_neighbors", &ObstacleGridParams::dropout_min_neighbors)
  .def_readwrite("empty_rays_hit_self", &ObstacleGridParams::empty_rays_hit_self)
  .def_readwrite("contamination_reach", &ObstacleGridParams::contamination_reach)
  .def_readwrite("unknown_contaminates_shadow", &ObstacleGridParams::unknown_contaminates_shadow)
  .def_readwrite("self_blind_range", &ObstacleGridParams::self_blind_range)
  .def_readwrite("static_after", &ObstacleGridParams::static_after)
  .def_readwrite("static_cell", &ObstacleGridParams::static_cell)
  .def_readwrite("static_gap", &ObstacleGridParams::static_gap)
  .def_readwrite("table_threads", &ObstacleGridParams::table_threads)
  .def_readwrite("threads", &ObstacleGridParams::threads)
  .def_readwrite("duplicate_stride", &ObstacleGridParams::duplicate_stride)
  .def_readwrite("cluster_radius", &ObstacleGridParams::cluster_radius)
  .def_readwrite("cluster_radius_per_m", &ObstacleGridParams::cluster_radius_per_m)
  .def_readwrite("min_cluster_size", &ObstacleGridParams::min_cluster_size)
  .def_readwrite("min_cluster_rings", &ObstacleGridParams::min_cluster_rings)
  .def_readwrite("veil_angle", &ObstacleGridParams::veil_angle)
  .def_readwrite("min_return_range", &ObstacleGridParams::min_return_range);

  py::enum_<Label>(m, "Label")
  .value("FREE", kFree)
  .value("OCCUPIED", kOccupied)
  .value("SELF", kSelf)
  .value("SHADOW", kShadow)
  .value("CONTAMINATED_SHADOW", kContaminatedShadow)
  .value("UNKNOWN", kUnknown)
  .value("STATIC", kStatic)
  .export_values();

  py::enum_<RayView>(m, "RayView")
  .value("NO_INFO", RayView::kNoInfo)
  .value("FREE", RayView::kFree)
  .value("HIT", RayView::kHit)
  .value("HIT_SELF", RayView::kHitSelf)
  .value("HIDDEN_BY_SELF", RayView::kHiddenBySelf)
  .value("HIDDEN_BY_OTHER", RayView::kHiddenByOther);

  py::class_<ObstacleGrid>(m, "ObstacleGrid")
  .def(py::init<const GridConfig &, const std::vector<LidarModel> &, const ObstacleGridParams &>(),
    py::arg("config"), py::arg("lidars"), py::arg("params") = ObstacleGridParams())
  .def_property_readonly("grid", &ObstacleGrid::grid, py::return_value_policy::reference_internal)
  .def("add_frame", [](ObstacleGrid & g, int lidar,
    py::array_t<float, py::array::c_style | py::array::forcecast> xyz,
    py::array_t<uint16_t, py::array::c_style | py::array::forcecast> ring,
    double stamp, const SelfModel & self) {
      if (xyz.ndim() != 2 || xyz.shape(1) != 3 || ring.size() != xyz.shape(0)) {
        throw std::invalid_argument("add_frame: xyz must be (n, 3) and ring (n,)");
      }
      py::gil_scoped_release release;
      g.add_frame(lidar, xyz.data(), ring.data(), static_cast<size_t>(xyz.shape(0)), stamp, self);
    }, py::arg("lidar"), py::arg("xyz"), py::arg("ring"), py::arg("stamp"), py::arg("self_model"))
  .def("set_static_self", &ObstacleGrid::set_static_self, py::arg("self_model"),
    "The robot parts that never move relative to the base, rasterized once.")
  .def("update", [](ObstacleGrid & g, const SelfModel & self, const Eigen::Matrix4f & world_from_base) {
      py::gil_scoped_release release;
      g.update(self, to_isometry(world_from_base));
    }, py::arg("self_model"), py::arg("world_from_base") = Eigen::Matrix4f::Identity(),
    "Relabel from the latest frames. world_from_base: base_footprint in odom, for static obstacles.")
  .def("labels", [](const ObstacleGrid & g) {
      const auto & labels = g.labels();
      return py::array_t<uint8_t>(static_cast<py::ssize_t>(labels.size()), labels.data());
    }, "A copy of the per-voxel labels (Label values).")
  .def("seen_by", [](const ObstacleGrid & g, const std::vector<size_t> & voxels) {
      py::array_t<uint8_t> out(static_cast<py::ssize_t>(voxels.size()));
      auto view = out.mutable_unchecked<1>();
      for (size_t i = 0; i < voxels.size(); ++i) {
        view(i) = g.seen_by(voxels[i]);
      }
      return out;
    }, py::arg("voxels"), "Per voxel: bit i set if LiDAR i's latest frame has a non-robot return in it.")
  .def("set_labels", [](ObstacleGrid & g, py::array_t<uint8_t, py::array::c_style | py::array::forcecast> labels) {
      std::vector<uint8_t> values(labels.data(), labels.data() + labels.size());
      py::gil_scoped_release release;
      g.set_labels(values);
    }, py::arg("labels"), "Labels from elsewhere (one per voxel) instead of LiDAR frames; rebuilds the tables.")
  .def("count_occupied", &ObstacleGrid::count_occupied, py::arg("box"))
  .def("count_unsafe", &ObstacleGrid::count_unsafe, py::arg("box"))
  .def("oldest_stamp", &ObstacleGrid::oldest_stamp)
  .def("ray_views", [](const ObstacleGrid & g, int lidar) {
      py::array_t<uint8_t> out(static_cast<py::ssize_t>(g.grid().size()));
      auto view = out.mutable_unchecked<1>();
      for (size_t i = 0; i < g.grid().size(); ++i) {
        view(i) = static_cast<uint8_t>(g.ray_view(lidar, i));
      }
      return out;
    }, py::arg("lidar"), "One LiDAR's RayView of every voxel, for debugging.")
  .def("range_image", [](const ObstacleGrid & g, int lidar) {
      const RangeImage & image = g.range_image(lidar);
      py::array_t<float> range({image.rows, image.cols});
      py::array_t<uint8_t> hit({image.rows, image.cols});
      std::copy(image.range.begin(), image.range.end(), range.mutable_data());
      std::transform(image.hit.begin(), image.hit.end(), hit.mutable_data(),
      [](Hit h) {return static_cast<uint8_t>(h);});
      return py::make_tuple(range, hit);
    }, py::arg("lidar"), "(range, hit) arrays of one LiDAR's latest range image.")
  .def("timing", [](const ObstacleGrid & g) {
      const Timing & t = g.timing();
      py::dict d;
      d["bin_points"] = t.bin_points;
      d["nearest"] = t.nearest;
      d["clusters"] = t.clusters;
      d["classify"] = t.classify;
      d["veil"] = t.veil;
      d["empty_rays"] = t.empty_rays;
      d["self_voxels"] = t.self_voxels;
      d["label"] = t.label;
      d["contaminate"] = t.contaminate;
      d["tables"] = t.tables;
      d["static_obstacles"] = t.static_obstacles;
      return d;
    }, "Milliseconds per step of the last add_frame and update.");

  py::class_<MotionLimits>(m, "MotionLimits")
  .def(py::init<>())
  .def_readwrite("contact_speed", &MotionLimits::contact_speed)
  .def_readwrite("unsafe_speed", &MotionLimits::unsafe_speed)
  .def_readwrite("escape_speed", &MotionLimits::escape_speed)
  .def_readwrite("max_escape_step", &MotionLimits::max_escape_step)
  .def_readwrite("standoff", &MotionLimits::standoff)
  .def_readwrite("obstacle_speed", &MotionLimits::obstacle_speed)
  .def_readwrite("unsafe_obstacle_speed", &MotionLimits::unsafe_obstacle_speed)
  .def_readwrite("latency", &MotionLimits::latency)
  .def_readwrite("deceleration", &MotionLimits::deceleration)
  .def_readwrite("max_speed", &MotionLimits::max_speed)
  .def_readwrite("max_piece", &MotionLimits::max_piece)
  .def_readwrite("max_clearance", &MotionLimits::max_clearance)
  .def_readwrite("clearance_resolution", &MotionLimits::clearance_resolution)
  .def_readwrite("min_motion", &MotionLimits::min_motion);

  py::class_<StepLimit>(m, "StepLimit")
  .def_readonly("min_duration", &StepLimit::min_duration)
  .def_readonly("speed", &StepLimit::speed)
  .def_readonly("travel", &StepLimit::travel)
  .def_readonly("clearance_occupied", &StepLimit::clearance_occupied)
  .def_readonly("clearance_moving", &StepLimit::clearance_moving)
  .def_readonly("clearance_unsafe", &StepLimit::clearance_unsafe)
  .def_readonly("blocked", &StepLimit::blocked)
  .def_readonly("unsafe", &StepLimit::unsafe)
  .def_readonly("escape", &StepLimit::escape)
  .def_readonly("shape", &StepLimit::shape);

  py::enum_<HitKind>(m, "HitKind")
  .value("OBSTACLES", HitKind::kObstacles)
  .value("UNSAFE", HitKind::kUnsafe);

  py::class_<StepHit>(m, "StepHit")
  .def_readonly("voxel", &StepHit::voxel)
  .def_readonly("shape", &StepHit::shape)
  .def_readonly("in_both", &StepHit::in_both);

  m.def("cylinder_bounds", &cylinder_bounds, py::arg("points"),
    "Bounds of a convex set of points (list of xyz) in cylindrical coordinates.");
  m.def("piece_corners", &piece_corners, py::arg("shape"), py::arg("f0") = Eigen::Vector3f::Zero(),
    py::arg("f1") = Eigen::Vector3f::Ones(),
    "Corners of the part of a shape between fractions f0 and f1 of its size along each axis.");
  m.def("grow", &grow, py::arg("box"), py::arg("d"));
  m.def("merge", &merge, py::arg("a"), py::arg("b"));
  m.def("allowed_speed", &allowed_speed, py::arg("clearance"), py::arg("final_speed"), py::arg("obstacle_speed"),
    py::arg("deceleration"), py::arg("limits"));

  py::class_<MotionCheck>(m, "MotionCheck")
  .def(py::init<const MotionLimits &>(), py::arg("limits") = MotionLimits())
  .def_property_readonly("limits", &MotionCheck::limits)
  .def("check", [](const MotionCheck & c, const ObstacleGrid & grid, const std::vector<SelfModel> & samples,
    float age, float theta_slack, const std::vector<float> & braking) {
      py::gil_scoped_release release;
      return c.check(grid, samples, age, theta_slack, braking);
    }, py::arg("grid"), py::arg("samples"), py::arg("age") = 0.0f, py::arg("theta_slack") = 0.0f,
    py::arg("braking") = std::vector<float>(),
    "Limits for each step between consecutive posed bodies (SelfModels) along a motion.")
  .def("first_blocked", [](const MotionCheck & c, const ObstacleGrid & grid, const std::vector<SelfModel> & samples,
    float theta_slack, float margin) {
      py::gil_scoped_release release;
      return c.first_blocked(grid, samples, theta_slack, margin);
    }, py::arg("grid"), py::arg("samples"), py::arg("theta_slack") = 0.0f, py::arg("margin") = 0.0f,
    "The first step check() would call blocked, or -1 (without clearances or speeds).")
  .def("blocked_at", [](const MotionCheck & c, const ObstacleGrid & grid, const SelfModel & body, float theta_slack,
    float margin) {
      py::gil_scoped_release release;
      return c.blocked_at(grid, body, theta_slack, margin);
    }, py::arg("grid"), py::arg("body"), py::arg("theta_slack") = 0.0f, py::arg("margin") = 0.0f,
    "True if the robot posed as `body` is as close to an obstacle as check() calls blocked.")
  .def("step_hits", [](const MotionCheck & c, const ObstacleGrid & grid, const std::vector<SelfModel> & samples,
    size_t step, HitKind kind, float theta_slack) {
      py::gil_scoped_release release;
      return c.step_hits(grid, samples, step, kind, theta_slack);
    }, py::arg("grid"), py::arg("samples"), py::arg("step"), py::arg("kind"), py::arg("theta_slack") = 0.0f,
    "The voxels behind one step of check() being blocked or unsafe: StepHits, by voxel.")
  .def("clearance_occupied", [](const MotionCheck & c, const ObstacleGrid & g, const CylinderBox & box) {
      return c.clearance(g.occupied_table(), box);
    }, py::arg("grid"), py::arg("box"))
  .def("clearance_unsafe", [](const MotionCheck & c, const ObstacleGrid & g, const CylinderBox & box) {
      return c.clearance(g.unsafe_table(), box);
    }, py::arg("grid"), py::arg("box"));
}
