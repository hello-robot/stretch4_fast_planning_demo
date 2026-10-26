#include "stretch_obstacle_grid/cylindrical_grid.hpp"

#include <cmath>
#include <stdexcept>

namespace stretch_obstacle_grid
{

namespace
{
constexpr float kTwoPi = 2.0f * static_cast<float>(M_PI);

float wrap_to_two_pi(float theta)
{
  theta = std::fmod(theta, kTwoPi);
  return theta < 0.0f ? theta + kTwoPi : theta;
}
}  // namespace

CylindricalGrid::CylindricalGrid(const GridConfig & config)
: config_(config)
{
  if (config.cell <= 0.0f || config.r_max <= config.r_min || config.z_max <= config.z_min) {
    throw std::invalid_argument("CylindricalGrid: bad extents");
  }
  radial_bins_ = static_cast<int>(std::lround((config.r_max - config.r_min) / config.cell));
  z_bins_ = static_cast<int>(std::lround((config.z_max - config.z_min) / config.cell));

  // Start with the power of two whose arc at r_min is closest to one cell, and double the
  // bins whenever a radial bin's outer arc would grow past two cells.
  int theta_bins = 1 << static_cast<int>(std::lround(std::log2(kTwoPi * config.r_min / config.cell)));
  band_index_.resize(radial_bins_);
  size_ = 0;
  for (int ri = 0; ri < radial_bins_; ++ri) {
    const float outer = config.r_min + (ri + 1) * config.cell;
    const bool start_band = bands_.empty() || kTwoPi * outer / theta_bins > 2.0f * config.cell + 1e-6f;
    if (start_band) {
      if (!bands_.empty()) {
        bands_.back().r_end = ri;
        theta_bins *= 2;
      }
      bands_.push_back(Band{theta_bins, kTwoPi / theta_bins, ri, ri, size_});
    }
    band_index_[ri] = static_cast<int>(bands_.size()) - 1;
    size_ += static_cast<size_t>(theta_bins) * z_bins_;
  }
  bands_.back().r_end = radial_bins_;
}

size_t CylindricalGrid::index(int ri, int ti, int zi) const
{
  const Band & band = band_of_radial(ri);
  return band.offset + (static_cast<size_t>(ri - band.r_begin) * band.theta_bins + ti) * z_bins_ + zi;
}

Cell CylindricalGrid::cell(size_t index) const
{
  // Bands are few (about five), so a linear scan is fine.
  size_t b = bands_.size() - 1;
  while (bands_[b].offset > index) {
    --b;
  }
  const Band & band = bands_[b];
  const size_t local = index - band.offset;
  const int zi = static_cast<int>(local % z_bins_);
  const size_t column = local / z_bins_;
  const int ti = static_cast<int>(column % band.theta_bins);
  const int ri = band.r_begin + static_cast<int>(column / band.theta_bins);
  return Cell{ri, ti, zi};
}

int CylindricalGrid::radial_bin(float r) const
{
  const int ri = static_cast<int>(std::floor((r - config_.r_min) / config_.cell));
  return (ri < 0 || ri >= radial_bins_) ? -1 : ri;
}

int CylindricalGrid::z_bin(float z) const
{
  const int zi = static_cast<int>(std::floor((z - config_.z_min) / config_.cell));
  return (zi < 0 || zi >= z_bins_) ? -1 : zi;
}

int CylindricalGrid::theta_bin(const Band & band, float theta)
{
  const int ti = static_cast<int>(wrap_to_two_pi(theta) / band.dtheta);
  return ti >= band.theta_bins ? band.theta_bins - 1 : ti;  // guard the 2pi rounding edge
}

bool CylindricalGrid::locate(float x, float y, float z, size_t * index) const
{
  const int ri = radial_bin(std::sqrt(x * x + y * y));
  const int zi = z_bin(z);
  if (ri < 0 || zi < 0) {
    return false;
  }
  *index = this->index(ri, theta_bin(band_of_radial(ri), std::atan2(y, x)), zi);
  return true;
}

void CylindricalGrid::bounds(
  const Cell & c, float * r0, float * r1, float * t0, float * t1, float * z0, float * z1) const
{
  const Band & band = band_of_radial(c.ri);
  *r0 = config_.r_min + c.ri * config_.cell;
  *r1 = *r0 + config_.cell;
  *t0 = c.ti * band.dtheta;
  *t1 = *t0 + band.dtheta;
  *z0 = config_.z_min + c.zi * config_.cell;
  *z1 = *z0 + config_.cell;
}

void CylindricalGrid::center(size_t index, float * x, float * y, float * z) const
{
  float r0, r1, t0, t1, z0, z1;
  bounds(cell(index), &r0, &r1, &t0, &t1, &z0, &z1);
  const float r = 0.5f * (r0 + r1);
  const float t = 0.5f * (t0 + t1);
  *x = r * std::cos(t);
  *y = r * std::sin(t);
  *z = 0.5f * (z0 + z1);
}

void CylindricalGrid::neighbors(size_t index, std::vector<size_t> * out) const
{
  const Cell c = cell(index);
  const Band & band = band_of_radial(c.ri);

  if (c.zi > 0) {out->push_back(index - 1);}
  if (c.zi + 1 < z_bins_) {out->push_back(index + 1);}

  const int n = band.theta_bins;
  out->push_back(this->index(c.ri, (c.ti + 1) % n, c.zi));
  out->push_back(this->index(c.ri, (c.ti + n - 1) % n, c.zi));

  // Radially: the same band shares theta bins; across a boundary the outer band has twice
  // as many, so a voxel faces two outer neighbors and one inner one.
  for (int dr : {-1, 1}) {
    const int rj = c.ri + dr;
    if (rj < 0 || rj >= radial_bins_) {
      continue;
    }
    const int other_bins = band_of_radial(rj).theta_bins;
    if (other_bins == n) {
      out->push_back(this->index(rj, c.ti, c.zi));
    } else if (other_bins == 2 * n) {
      out->push_back(this->index(rj, 2 * c.ti, c.zi));
      out->push_back(this->index(rj, 2 * c.ti + 1, c.zi));
    } else {
      out->push_back(this->index(rj, c.ti / 2, c.zi));
    }
  }
}

}  // namespace stretch_obstacle_grid
