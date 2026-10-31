#include "btkernel/volume_srq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#if defined(BTKERNEL_HAS_OPENMP)
#include <omp.h>
#endif

namespace btkernel {
namespace {

constexpr double kTiny = 1.0e-300;
constexpr double kPi = 3.141592653589793238462643383279502884;

double norm2(const std::complex<double>* x, const std::size_t size) {
  double sum = 0.0;
  for (std::size_t idx = 0; idx < size; ++idx) {
    sum += std::norm(x[idx]);
  }
  return sum;
}

double real_inner(const std::complex<double>* x,
                  const std::complex<double>* y,
                  const std::size_t size) {
  std::complex<double> sum{0.0, 0.0};
  for (std::size_t idx = 0; idx < size; ++idx) {
    sum += std::conj(x[idx]) * y[idx];
  }
  return sum.real();
}

void zero(std::complex<double>* x, const std::size_t size) {
  std::fill(x, x + size, std::complex<double>{0.0, 0.0});
}

void copy_buffer(const std::complex<double>* src, std::complex<double>* dst, const std::size_t size) {
  std::copy(src, src + size, dst);
}

double choose_lambda_bound(const BTKernel& kernel, const double requested_lambda) {
  if (!std::isfinite(requested_lambda)) {
    throw std::invalid_argument("VolumeSRQ requires a finite Lambda bound or automatic Lambda selection.");
  }
  if (requested_lambda >= 0.0) {
    return requested_lambda;
  }
  const double bound = kernel.row_sum_bound();
  if (bound <= 0.0) {
    return 0.0;
  }
  return bound * (1.0 + 64.0 * std::numeric_limits<double>::epsilon());
}

}

SRQGrid make_srq_grid(const int shifts) {
  if (shifts <= 0) {
    throw std::invalid_argument("SRQ shifts must be positive.");
  }

  SRQGrid grid;
  grid.shifts = shifts;
  grid.kminus = static_cast<int>(std::ceil(3.0 * static_cast<double>(shifts - 1) / 4.0));
  grid.kplus = shifts - 1 - grid.kminus;
  grid.h = 4.0 * kPi / std::sqrt(3.0 * static_cast<double>(shifts));
  grid.taus.reserve(static_cast<std::size_t>(shifts));
  grid.weights.reserve(static_cast<std::size_t>(shifts));
  for (int k = -grid.kminus; k <= grid.kplus; ++k) {
    const double kh = static_cast<double>(k) * grid.h;
    grid.taus.push_back(std::exp(kh));
    grid.weights.push_back(std::exp(0.25 * kh));
  }
  return grid;
}

VolumeSRQ::VolumeSRQ(std::vector<Spin> two_js,
                     std::vector<TripleSign> triple_signs,
                     const double lambda_bound,
                     const int shifts,
                     const double cg_tol,
                     const int cg_maxiter,
                     std::vector<Channel> channels)
    : kernel_(std::move(two_js), std::move(triple_signs), std::move(channels)),
      lambda_bound_(choose_lambda_bound(kernel_, lambda_bound)),
      lambda_sq_(lambda_bound_ * lambda_bound_),
      grid_(make_srq_grid(shifts)),
      cg_tol_(cg_tol),
      cg_maxiter_(cg_maxiter > 0 ? cg_maxiter : default_maxiter()) {
  if (!std::isfinite(lambda_bound_) || lambda_bound_ < 0.0) {
    throw std::invalid_argument("VolumeSRQ requires a finite nonnegative Lambda bound.");
  }
  if (!std::isfinite(cg_tol_) || cg_tol_ <= 0.0) {
    throw std::invalid_argument("VolumeSRQ requires a positive CG tolerance.");
  }
  initialize_four_valent_bands();
  ensure_work(work_);
}

int VolumeSRQ::default_maxiter() const {
  const std::size_t d = kernel_.dim();
  if (d == 0) {
    return 8;
  }
  const std::size_t exact_small_sector = 2 * d + 4;
  const std::size_t value = (d <= 512) ? exact_small_sector : 512;
  return static_cast<int>(std::min<std::size_t>(value, static_cast<std::size_t>(std::numeric_limits<int>::max())));
}

void VolumeSRQ::validate_vector_size(const std::size_t size, const char* caller) const {
  if (size != dim()) {
    throw std::invalid_argument(std::string(caller) + " dimension mismatch.");
  }
}

void VolumeSRQ::ensure_work(WorkBuffers& work) const {
  const std::size_t d = dim();
  work.q_tmp.assign(d, {0.0, 0.0});
  work.qq_tmp.assign(d, {0.0, 0.0});
  work.rhs.assign(d, {0.0, 0.0});
  work.x.assign(d, {0.0, 0.0});
  work.r.assign(d, {0.0, 0.0});
  work.p.assign(d, {0.0, 0.0});
  work.ap.assign(d, {0.0, 0.0});
  work.shifted_ap.assign(d, {0.0, 0.0});
  const std::size_t krylov_max = static_cast<std::size_t>(std::max(cg_maxiter_, 1));
  work.lanczos_alpha.assign(krylov_max, 0.0);
  work.lanczos_beta.assign(krylov_max, 0.0);
  work.tridiagonal_cprime.assign(krylov_max, 0.0);
  work.tridiagonal_solution.assign(krylov_max, 0.0);
  work.combined_coefficients.assign(krylov_max, 0.0);
  work.relres_shift.assign(static_cast<std::size_t>(grid_.shifts), 0.0);
  work.banded_rhs.assign(d, {0.0, 0.0});
  work.banded_acc.assign(d, {0.0, 0.0});
  work.banded_dprime.assign(d, {0.0, 0.0});
  work.banded_cprime.assign(d, 0.0);
}

void VolumeSRQ::initialize_four_valent_bands() {
  four_valent_banded_ = false;
  four_abar_diag_.clear();
  four_abar_off2_.clear();
  if (!kernel_.four_valent_fast_path() || lambda_bound_ <= 0.0) {
    return;
  }

  const auto& links = kernel_.four_valent_links();
  const std::size_t d = kernel_.dim();
  if (d == 0) {
    return;
  }
  four_abar_diag_.assign(d, 0.0);
  four_abar_off2_.assign(d >= 2 ? d - 2 : 0, 0.0);

  const double sigma = static_cast<double>(kernel_.four_valent_sigma());
  const double scale = (sigma * sigma) / lambda_sq_;
  if (scale == 0.0) {
    four_valent_banded_ = true;
    return;
  }

  for (std::size_t link = 0; link < links.size(); ++link) {
    const double value = links[link];
    const double squared = scale * value * value;
    four_abar_diag_[link] += squared;
    four_abar_diag_[link + 1] += squared;
  }
  if (links.size() >= 2) {
    for (std::size_t idx = 0; idx + 1 < links.size(); ++idx) {
      four_abar_off2_[idx] = -scale * links[idx] * links[idx + 1];
    }
  }
  four_valent_banded_ = true;
}

void VolumeSRQ::apply_four_valent_abar(const std::complex<double>* input,
                                       std::complex<double>* output) const {
  const std::size_t d = dim();
  if (lambda_bound_ <= 0.0 || !four_valent_banded_) {
    zero(output, d);
    return;
  }

#if defined(BTKERNEL_HAS_OPENMP)
#pragma omp parallel for schedule(static) if (d >= 8192)
#endif
  for (std::ptrdiff_t idx_signed = 0; idx_signed < static_cast<std::ptrdiff_t>(d); ++idx_signed) {
    const std::size_t idx = static_cast<std::size_t>(idx_signed);
    std::complex<double> value = four_abar_diag_[idx] * input[idx];
    if (idx >= 2) {
      value += four_abar_off2_[idx - 2] * input[idx - 2];
    }
    if (idx + 2 < d) {
      value += four_abar_off2_[idx] * input[idx + 2];
    }
    output[idx] = value;
  }
}

void VolumeSRQ::solve_four_valent_shift(const double tau,
                                        const std::complex<double>* rhs,
                                        std::complex<double>* solution,
                                        WorkBuffers& work) const {
  const std::size_t d = dim();
  zero(solution, d);
  if (d == 0) {
    return;
  }

  for (std::size_t parity = 0; parity < 2; ++parity) {
    if (parity >= d) {
      continue;
    }
    const std::size_t count = 1 + (d - 1 - parity) / 2;
    if (count == 0) {
      continue;
    }

    const std::size_t first = parity;
    double denom = four_abar_diag_[first] + tau;
    if (std::abs(denom) <= kTiny) {
      denom = (denom < 0.0) ? -kTiny : kTiny;
    }
    work.banded_cprime[0] = (count > 1) ? four_abar_off2_[first] / denom : 0.0;
    work.banded_dprime[first] = rhs[first] / denom;

    for (std::size_t local = 1; local < count; ++local) {
      const std::size_t idx = parity + 2 * local;
      const std::size_t prev = idx - 2;
      const double off_prev = four_abar_off2_[prev];
      denom = four_abar_diag_[idx] + tau - off_prev * work.banded_cprime[local - 1];
      if (std::abs(denom) <= kTiny) {
        denom = (denom < 0.0) ? -kTiny : kTiny;
      }
      work.banded_cprime[local] =
          (local + 1 < count && idx < four_abar_off2_.size()) ? four_abar_off2_[idx] / denom : 0.0;
      work.banded_dprime[idx] = (rhs[idx] - off_prev * work.banded_dprime[prev]) / denom;
    }

    const std::size_t last = parity + 2 * (count - 1);
    solution[last] = work.banded_dprime[last];
    for (std::size_t local = count - 1; local-- > 0;) {
      const std::size_t idx = parity + 2 * local;
      const std::size_t next = idx + 2;
      solution[idx] = work.banded_dprime[idx] - work.banded_cprime[local] * solution[next];
    }
  }
}

void VolumeSRQ::apply_four_valent_srq(const std::complex<double>* input,
                                      std::complex<double>* output,
                                      WorkBuffers& work) {
  const std::size_t d = dim();
  zero(output, d);
  if (lambda_bound_ <= 0.0 || d == 0) {
    last_diagnostics_ = {};
    return;
  }

  apply_four_valent_abar(input, work.banded_rhs.data());
  const double rhs_norm2 = norm2(work.banded_rhs.data(), d);
  if (rhs_norm2 <= kTiny) {
    last_diagnostics_ = {};
    last_diagnostics_.shifted_solves = grid_.shifts;
    return;
  }

  zero(work.banded_acc.data(), d);
  for (int shift = 0; shift < grid_.shifts; ++shift) {
    const std::size_t s = static_cast<std::size_t>(shift);
    solve_four_valent_shift(grid_.taus[s], work.banded_rhs.data(), work.x.data(), work);
    const double weight = grid_.weights[s];
#if defined(BTKERNEL_HAS_OPENMP)
#pragma omp parallel for schedule(static) if (d >= 8192)
#endif
    for (std::ptrdiff_t idx_signed = 0; idx_signed < static_cast<std::ptrdiff_t>(d); ++idx_signed) {
      const std::size_t idx = static_cast<std::size_t>(idx_signed);
      work.banded_acc[idx] += weight * work.x[idx];
    }
  }

  const double prefactor = std::sqrt(lambda_bound_) * grid_.h / (std::sqrt(2.0) * kPi);
#if defined(BTKERNEL_HAS_OPENMP)
#pragma omp parallel for schedule(static) if (d >= 8192)
#endif
  for (std::ptrdiff_t idx_signed = 0; idx_signed < static_cast<std::ptrdiff_t>(d); ++idx_signed) {
    const std::size_t idx = static_cast<std::size_t>(idx_signed);
    output[idx] = prefactor * work.banded_acc[idx];
  }

  last_diagnostics_ = {};
  last_diagnostics_.max_relres = 0.0;
  last_diagnostics_.mean_relres = 0.0;
  last_diagnostics_.shifted_solves = grid_.shifts;
  last_diagnostics_.cg_iterations = 0;
  last_diagnostics_.converged = true;
}

void VolumeSRQ::apply_abar_with_work(const std::complex<double>* input,
                                     std::complex<double>* output,
                                     WorkBuffers& work,
                                     const bool mpi) {
  if (lambda_bound_ <= 0.0) {
    zero(output, dim());
    return;
  }

  if (mpi) {
    kernel_.apply_into_mpi_world(input, work.q_tmp.data(), dim());
    kernel_.apply_into_mpi_world(work.q_tmp.data(), work.qq_tmp.data(), dim());
  } else {
    kernel_.apply_into(input, work.q_tmp.data(), dim());
    kernel_.apply_into(work.q_tmp.data(), work.qq_tmp.data(), dim());
  }

  const double scale = 1.0 / lambda_sq_;
  for (std::size_t idx = 0; idx < dim(); ++idx) {
    output[idx] = scale * work.qq_tmp[idx];
  }
}

void VolumeSRQ::apply_abar_into(const std::complex<double>* input,
                                std::complex<double>* output,
                                const std::size_t size) {
  validate_vector_size(size, "VolumeSRQ::apply_abar_into");
  if (input == nullptr || output == nullptr) {
    throw std::invalid_argument("VolumeSRQ::apply_abar_into received a null buffer.");
  }
  apply_abar_with_work(input, output, work_, false);
}

VolumeSRQ::SolveStats VolumeSRQ::solve_shifted(const double tau,
                                               const std::complex<double>* rhs,
                                               std::complex<double>* solution,
                                               WorkBuffers& work,
                                               const bool mpi) {
  const std::size_t d = dim();
  SolveStats stats;
  zero(solution, d);
  copy_buffer(rhs, work.r.data(), d);
  copy_buffer(rhs, work.p.data(), d);

  const double rhs_norm2 = norm2(rhs, d);
  if (rhs_norm2 <= kTiny) {
    zero(solution, d);
    stats.relres = 0.0;
    stats.iterations = 0;
    stats.converged = true;
    return stats;
  }

  const double threshold = cg_tol_ * cg_tol_ * rhs_norm2;
  double rs = rhs_norm2;
  stats.converged = false;

  for (int iter = 0; iter < cg_maxiter_; ++iter) {
    apply_abar_with_work(work.p.data(), work.ap.data(), work, mpi);
    for (std::size_t idx = 0; idx < d; ++idx) {
      work.shifted_ap[idx] = work.ap[idx] + tau * work.p[idx];
    }

    const double denom = real_inner(work.p.data(), work.shifted_ap.data(), d);
    if (std::abs(denom) <= kTiny) {
      break;
    }
    const double alpha = rs / denom;
    for (std::size_t idx = 0; idx < d; ++idx) {
      solution[idx] += alpha * work.p[idx];
      work.r[idx] -= alpha * work.shifted_ap[idx];
    }

    const double rs_next = norm2(work.r.data(), d);
    stats.iterations = iter + 1;
    if (rs_next <= threshold) {
      rs = rs_next;
      stats.converged = true;
      break;
    }
    const double beta = rs_next / std::max(rs, kTiny);
    for (std::size_t idx = 0; idx < d; ++idx) {
      work.p[idx] = work.r[idx] + beta * work.p[idx];
    }
    rs = rs_next;
  }

  stats.relres = std::sqrt(std::max(rs, 0.0) / std::max(rhs_norm2, kTiny));
  if (stats.relres <= cg_tol_) {
    stats.converged = true;
  }
  return stats;
}

SRQDiagnostics VolumeSRQ::solve_shifted_sum_lanczos(const std::complex<double>* rhs,
                                                    std::complex<double>* weighted_sum,
                                                    WorkBuffers& work,
                                                    const bool mpi) {
  const std::size_t d = dim();
  zero(weighted_sum, d);

  SRQDiagnostics diagnostics;
  diagnostics.shifted_solves = grid_.shifts;
  diagnostics.converged = true;

  const double rhs_norm2 = norm2(rhs, d);
  if (rhs_norm2 <= kTiny) {
    return diagnostics;
  }

  const double rhs_norm = std::sqrt(rhs_norm2);
  const std::size_t max_krylov =
      std::min<std::size_t>(static_cast<std::size_t>(std::max(cg_maxiter_, 1)), d);
  const double breakdown_tol = 64.0 * std::numeric_limits<double>::epsilon();

  std::fill(work.lanczos_alpha.begin(), work.lanczos_alpha.end(), 0.0);
  std::fill(work.lanczos_beta.begin(), work.lanczos_beta.end(), 0.0);
  std::fill(work.combined_coefficients.begin(), work.combined_coefficients.end(), 0.0);
  std::fill(work.relres_shift.begin(), work.relres_shift.end(), 1.0);

  auto solve_projected_shift = [&](const std::size_t m,
                                   const double tau,
                                   std::vector<double>& solution) {
    auto& cprime = work.tridiagonal_cprime;
    std::fill(solution.begin(), solution.begin() + static_cast<std::ptrdiff_t>(m), 0.0);
    std::fill(cprime.begin(), cprime.begin() + static_cast<std::ptrdiff_t>(m), 0.0);

    double denom = work.lanczos_alpha[0] + tau;
    if (std::abs(denom) <= kTiny) {
      denom = (denom < 0.0) ? -kTiny : kTiny;
    }
    if (m > 1) {
      cprime[0] = work.lanczos_beta[0] / denom;
    }
    solution[0] = rhs_norm / denom;

    for (std::size_t row = 1; row < m; ++row) {
      denom = work.lanczos_alpha[row] + tau - work.lanczos_beta[row - 1] * cprime[row - 1];
      if (std::abs(denom) <= kTiny) {
        denom = (denom < 0.0) ? -kTiny : kTiny;
      }
      if (row + 1 < m) {
        cprime[row] = work.lanczos_beta[row] / denom;
      }
      solution[row] = (-work.lanczos_beta[row - 1] * solution[row - 1]) / denom;
    }

    for (std::size_t row = m - 1; row-- > 0;) {
      solution[row] -= cprime[row] * solution[row + 1];
    }
  };

  zero(work.r.data(), d);
  for (std::size_t idx = 0; idx < d; ++idx) {
    work.p[idx] = rhs[idx] / rhs_norm;
  }

  std::size_t krylov_dim = 0;
  bool exact_breakdown = false;

  for (std::size_t iter = 0; iter < max_krylov; ++iter) {
    apply_abar_with_work(work.p.data(), work.ap.data(), work, mpi);
    if (iter > 0) {
      const double beta_previous = work.lanczos_beta[iter - 1];
      for (std::size_t idx = 0; idx < d; ++idx) {
        work.ap[idx] -= beta_previous * work.r[idx];
      }
    }

    const double alpha = real_inner(work.p.data(), work.ap.data(), d);
    work.lanczos_alpha[iter] = alpha;
    for (std::size_t idx = 0; idx < d; ++idx) {
      work.ap[idx] -= alpha * work.p[idx];
    }

    const double beta = std::sqrt(std::max(norm2(work.ap.data(), d), 0.0));
    work.lanczos_beta[iter] = beta;
    krylov_dim = iter + 1;

    const bool breakdown = beta <= breakdown_tol;
    const bool reached_limit = iter + 1 == max_krylov;
    const bool check_residual = breakdown || reached_limit || ((iter + 1) % 8 == 0);
    if (check_residual) {
      double max_relres = 0.0;
      for (int shift = 0; shift < grid_.shifts; ++shift) {
        solve_projected_shift(krylov_dim,
                              grid_.taus[static_cast<std::size_t>(shift)],
                              work.tridiagonal_solution);
        const double relres =
            std::abs(beta * work.tridiagonal_solution[krylov_dim - 1]) / std::max(rhs_norm, kTiny);
        max_relres = std::max(max_relres, relres);
      }
      if (breakdown || reached_limit || max_relres <= cg_tol_) {
        exact_breakdown = breakdown;
        break;
      }
    }

    if (breakdown || reached_limit) {
      exact_breakdown = breakdown;
      break;
    }

    copy_buffer(work.p.data(), work.shifted_ap.data(), d);
    for (std::size_t idx = 0; idx < d; ++idx) {
      work.p[idx] = work.ap[idx] / beta;
      work.r[idx] = work.shifted_ap[idx];
    }
  }

  if (krylov_dim == 0) {
    return diagnostics;
  }

  diagnostics.max_relres = 0.0;
  diagnostics.mean_relres = 0.0;
  // the projected residual norm is beta_m times the last solution coefficient
  const double residual_beta = exact_breakdown ? 0.0 : work.lanczos_beta[krylov_dim - 1];
  for (int shift = 0; shift < grid_.shifts; ++shift) {
    const std::size_t s = static_cast<std::size_t>(shift);
    solve_projected_shift(krylov_dim, grid_.taus[s], work.tridiagonal_solution);
    const double weight = grid_.weights[s];
    for (std::size_t row = 0; row < krylov_dim; ++row) {
      work.combined_coefficients[row] += weight * work.tridiagonal_solution[row];
    }
    const double relres = std::abs(residual_beta * work.tridiagonal_solution[krylov_dim - 1]) /
                          std::max(rhs_norm, kTiny);
    work.relres_shift[s] = relres;
    diagnostics.max_relres = std::max(diagnostics.max_relres, relres);
    diagnostics.mean_relres += relres;
  }
  diagnostics.mean_relres /= static_cast<double>(std::max(grid_.shifts, 1));
  diagnostics.cg_iterations = static_cast<std::int64_t>(krylov_dim);
  diagnostics.converged = diagnostics.max_relres <= cg_tol_;

  zero(work.r.data(), d);
  for (std::size_t idx = 0; idx < d; ++idx) {
    work.p[idx] = rhs[idx] / rhs_norm;
  }

  // replay the recurrence instead of storing every Lanczos vector
  for (std::size_t iter = 0; iter < krylov_dim; ++iter) {
    const double coeff = work.combined_coefficients[iter];
#if defined(BTKERNEL_HAS_OPENMP)
#pragma omp parallel for schedule(static) if (d >= 8192)
#endif
    for (std::ptrdiff_t idx_signed = 0; idx_signed < static_cast<std::ptrdiff_t>(d); ++idx_signed) {
      const std::size_t idx = static_cast<std::size_t>(idx_signed);
      weighted_sum[idx] += coeff * work.p[idx];
    }

    if (iter + 1 == krylov_dim) {
      break;
    }

    apply_abar_with_work(work.p.data(), work.ap.data(), work, mpi);
    if (iter > 0) {
      const double beta_previous = work.lanczos_beta[iter - 1];
      for (std::size_t idx = 0; idx < d; ++idx) {
        work.ap[idx] -= beta_previous * work.r[idx];
      }
    }
    for (std::size_t idx = 0; idx < d; ++idx) {
      work.ap[idx] -= work.lanczos_alpha[iter] * work.p[idx];
    }
    const double beta = work.lanczos_beta[iter];
    if (beta <= breakdown_tol) {
      break;
    }
    copy_buffer(work.p.data(), work.shifted_ap.data(), d);
    for (std::size_t idx = 0; idx < d; ++idx) {
      work.p[idx] = work.ap[idx] / beta;
      work.r[idx] = work.shifted_ap[idx];
    }
  }
  return diagnostics;
}

void VolumeSRQ::apply_impl(const std::complex<double>* input,
                           std::complex<double>* output,
                           const bool mpi) {
  if (lambda_bound_ <= 0.0 || dim() == 0) {
    zero(output, dim());
    last_diagnostics_ = {};
    return;
  }

  // the Q squared rhs keeps exact kernel components out of the shifted solves
  apply_abar_with_work(input, work_.rhs.data(), work_, mpi);
  const auto& rhs = work_.rhs;
  const double rhs_norm2 = norm2(rhs.data(), dim());
  if (rhs_norm2 <= kTiny) {
    zero(output, dim());
    last_diagnostics_ = {};
    last_diagnostics_.shifted_solves = grid_.shifts;
    return;
  }

  const double prefactor = std::sqrt(lambda_bound_) * grid_.h / (std::sqrt(2.0) * kPi);
  SRQDiagnostics diagnostics = solve_shifted_sum_lanczos(rhs.data(), output, work_, mpi);
  for (std::size_t idx = 0; idx < dim(); ++idx) {
    output[idx] *= prefactor;
  }
  last_diagnostics_ = diagnostics;
}

void VolumeSRQ::apply_into(const std::complex<double>* input,
                           std::complex<double>* output,
                           const std::size_t size) {
  validate_vector_size(size, "VolumeSRQ::apply_into");
  if (input == nullptr || output == nullptr) {
    throw std::invalid_argument("VolumeSRQ::apply_into received a null buffer.");
  }
  apply_impl(input, output, false);
}

void VolumeSRQ::apply_into_mpi_world(const std::complex<double>* input,
                                     std::complex<double>* output,
                                     const std::size_t size) {
  validate_vector_size(size, "VolumeSRQ::apply_into_mpi_world");
  if (input == nullptr || output == nullptr) {
    throw std::invalid_argument("VolumeSRQ::apply_into_mpi_world received a null buffer.");
  }
  apply_impl(input, output, true);
}

ComplexVector VolumeSRQ::apply(const ComplexVector& input) {
  validate_vector_size(input.size(), "VolumeSRQ::apply");
  ComplexVector out(dim(), {0.0, 0.0});
  apply_into(input.data(), out.data(), input.size());
  return out;
}

}
