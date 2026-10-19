// Python bindings: `import stretch_obstacle_grid`.

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
  .def_readwrite("z_max", &CylinderBox::z_max);

  py::class_<LidarModel>(m, "LidarModel")
  .def(py::init([](const Eigen::Matrix4f & base_from_lidar, std::vector<float> ring_elevation, int azimuth_bins) {
      return LidarModel{to_isometry(base_from_lidar), std::move(ring_elevation), azimuth_bins};
    }), py::arg("base_from_lidar"), py::arg("ring_elevation"), py::arg("azimuth_bins") = 720)
  .def_readonly("ring_elevation", &LidarModel::ring_elevation)
  .def_readonly("azimuth_bins", &LidarModel::azimuth_bins);

  py::class_<SelfShape>(m, "SelfShape")
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
  .def("ray_distance", &SelfModel::ray_distance, py::arg("origin"), py::arg("direction"),
    "Distance along a ray (unit direction) to the first shape it enters from outside, or inf.");

  py::class_<ObstacleGridParams>(m, "ObstacleGridParams")
  .def(py::init<>())
  .def_readwrite("self_point_padding", &ObstacleGridParams::self_point_padding)
  .def_readwrite("self_voxel_padding", &ObstacleGridParams::self_voxel_padding)
  .def_readwrite("range_margin", &ObstacleGridParams::range_margin)
  .def_readwrite("dropout_min_neighbors", &ObstacleGridParams::dropout_min_neighbors)
  .def_readwrite("empty_rays_hit_self", &ObstacleGridParams::empty_rays_hit_self)
  .def_readwrite("contamination_reach", &ObstacleGridParams::contamination_reach)
  .def_readwrite("table_threads", &ObstacleGridParams::table_threads)
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
  .def("update", [](ObstacleGrid & g, const SelfModel & self) {
      py::gil_scoped_release release;
      g.update(self);
    }, py::arg("self_model"))
  .def("labels", [](const ObstacleGrid & g) {
      const auto & labels = g.labels();
      return py::array_t<uint8_t>(static_cast<py::ssize_t>(labels.size()), labels.data());
    }, "A copy of the per-voxel labels (Label values).")
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
      d["self_voxels"] = t.self_voxels;
      d["label"] = t.label;
      d["contaminate"] = t.contaminate;
      d["tables"] = t.tables;
      return d;
    }, "Milliseconds per step of the last add_frame and update.");
}
