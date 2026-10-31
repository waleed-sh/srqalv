#include "btkernel/wigner6j.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace btkernel {
namespace {

double delta_factor(const Spin a, const Spin b, const Spin c) {
  if (!triangle_ok(a, b, c)) {
    return 0.0;
  }
  const int x = (a + b - c) / 2;
  const int y = (a - b + c) / 2;
  const int z = (-a + b + c) / 2;
  const int w = (a + b + c) / 2 + 1;
  return std::exp(0.5 * (std::lgamma(x + 1.0) + std::lgamma(y + 1.0) +
                         std::lgamma(z + 1.0) - std::lgamma(w + 1.0)));
}

}

bool triangle_ok(const Spin a, const Spin b, const Spin c) noexcept {
  return std::abs(a - b) <= c && c <= a + b && ((a + b + c) % 2 == 0);
}

double Wigner6jCache::get(const Spin j1,
                          const Spin j2,
                          const Spin j3,
                          const Spin l1,
                          const Spin l2,
                          const Spin l3) {
  const Key key{{j1, j2, j3, l1, l2, l3}};
  const auto found = cache_.find(key);
  if (found != cache_.end()) {
    return found->second;
  }
  const double value = compute(j1, j2, j3, l1, l2, l3);
  cache_.emplace(key, value);
  return value;
}

double Wigner6jCache::compute(const Spin j1,
                              const Spin j2,
                              const Spin j3,
                              const Spin l1,
                              const Spin l2,
                              const Spin l3) {
  const double prefactor = delta_factor(j1, j2, j3) * delta_factor(j1, l2, l3) *
                           delta_factor(l1, j2, l3) * delta_factor(l1, l2, j3);
  if (prefactor == 0.0) {
    return 0.0;
  }

  const int sums[7] = {
      j1 + j2 + j3,
      j1 + l2 + l3,
      l1 + j2 + l3,
      l1 + l2 + j3,
      j1 + j2 + l1 + l2,
      j1 + j3 + l1 + l3,
      j2 + j3 + l2 + l3,
  };
  for (const int total : sums) {
    if (total % 2 != 0) {
      return 0.0;
    }
  }

  const int a1 = sums[0] / 2;
  const int a2 = sums[1] / 2;
  const int a3 = sums[2] / 2;
  const int a4 = sums[3] / 2;
  const int b1 = sums[4] / 2;
  const int b2 = sums[5] / 2;
  const int b3 = sums[6] / 2;
  const int z_min = std::max(std::max(a1, a2), std::max(a3, a4));
  const int z_max = std::min(std::min(b1, b2), b3);
  if (z_min > z_max) {
    return 0.0;
  }

  double total = 0.0;
  for (int z = z_min; z <= z_max; ++z) {
    const int den_args[7] = {z - a1, z - a2, z - a3, z - a4, b1 - z, b2 - z, b3 - z};
    bool valid = true;
    double log_den = 0.0;
    for (const int arg : den_args) {
      if (arg < 0) {
        valid = false;
        break;
      }
      log_den += std::lgamma(arg + 1.0);
    }
    if (!valid) {
      continue;
    }
    const double log_term = std::lgamma(z + 2.0) - log_den;
    total += (z % 2 == 0 ? 1.0 : -1.0) * std::exp(log_term);
  }
  return prefactor * total;
}

}

