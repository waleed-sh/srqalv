#include "btkernel/bt_kernel.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(BTKERNEL_HAS_OPENMP)
#include <omp.h>
#endif

#if defined(BTKERNEL_HAS_MPI)
#include <mpi.h>
#endif

namespace btkernel {
namespace {

constexpr double kDropTol = 1.0e-14;

double phase_from_twice(const int total_twice) {
  if (total_twice % 2 != 0) {
    return 0.0;
  }
  return ((total_twice / 2) % 2 == 0) ? 1.0 : -1.0;
}

double sqrt_dim(const Spin a, const Spin b) {
  return std::sqrt((static_cast<double>(a) + 1.0) * (static_cast<double>(b) + 1.0));
}

double x_factor(const Spin two_a, const Spin two_b) {
  const double a = static_cast<double>(two_a);
  const double b = static_cast<double>(two_b);
  return a * (a + 1.0) * (a + 2.0) * b * (b + 1.0) * (b + 2.0);
}

double casimir(const Spin two_a) {
  const double a = static_cast<double>(two_a);
  return a * (a + 2.0) / 4.0;
}

double four_valent_offdiag_raw(const Spin two_j1,
                               const Spin two_j2,
                               const Spin two_j3,
                               const Spin two_j4,
                               const Spin upper_channel) {
  const double j1 = static_cast<double>(two_j1);
  const double j2 = static_cast<double>(two_j2);
  const double j3 = static_cast<double>(two_j3);
  const double j4 = static_cast<double>(two_j4);
  const double x = static_cast<double>(upper_channel);
  if (x <= 1.0) {
    return 0.0;
  }
  const double left = (x + j1 + j2 + 2.0) * (-j1 + j2 + x) * (j1 - j2 + x) *
                      (j1 + j2 - x + 2.0);
  const double right = (x + j3 + j4 + 2.0) * (-j3 + j4 + x) * (j3 - j4 + x) *
                       (j3 + j4 - x + 2.0);
  const double product = left * right;
  if (product <= 1.0e-28) {
    return 0.0;
  }
  return std::sqrt(product) / (16.0 * std::sqrt((x - 1.0) * (x + 1.0)));
}

int four_valent_closure_sign(const TripleSign& triple) {
  if (triple.i == 0 && triple.j == 1 && triple.k == 2) {
    return 1;
  }
  if (triple.i == 0 && triple.j == 1 && triple.k == 3) {
    return -1;
  }
  if (triple.i == 0 && triple.j == 2 && triple.k == 3) {
    return 1;
  }
  if (triple.i == 1 && triple.j == 2 && triple.k == 3) {
    return -1;
  }
  return 0;
}

std::vector<std::string> split(const std::string& text, const char delim) {
  std::vector<std::string> parts;
  std::stringstream stream(text);
  std::string part;
  while (std::getline(stream, part, delim)) {
    if (!part.empty()) {
      parts.push_back(part);
    }
  }
  return parts;
}

int requested_openmp_threads(const std::size_t work_items) noexcept {
#if defined(BTKERNEL_HAS_OPENMP)
  const int max_threads = omp_get_max_threads();
  if (max_threads <= 1 || work_items < 64) {
    return 1;
  }
  const std::size_t by_work = std::max<std::size_t>(1, work_items / 64);
  return std::max(1, std::min(max_threads, static_cast<int>(by_work)));
#else
  (void) work_items;
  return 1;
#endif
}

#if defined(BTKERNEL_HAS_MPI)
void mpi_allreduce_complex_sum(const std::complex<double>* local,
                               std::complex<double>* global,
                               const std::size_t size) {
  const auto* local_raw = reinterpret_cast<const double*>(local);
  auto* global_raw = reinterpret_cast<double*>(global);
  std::size_t offset = 0;
  const std::size_t total = 2 * size;
  constexpr int kChunk = 1 << 26;
  while (offset < total) {
    const int count = static_cast<int>(std::min<std::size_t>(kChunk, total - offset));
    MPI_Allreduce(local_raw + offset, global_raw + offset, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    offset += static_cast<std::size_t>(count);
  }
}
#endif

}

double four_valent_offdiag(const std::vector<Spin>& two_js4, const Spin upper_channel) {
  if (two_js4.size() != 4) {
    throw std::invalid_argument("four_valent_offdiag requires four incident spins.");
  }
  return four_valent_offdiag_raw(two_js4[0], two_js4[1], two_js4[2], two_js4[3], upper_channel);
}

BTKernel::BTKernel(std::vector<Spin> two_js,
                   std::vector<TripleSign> triple_signs,
                   std::vector<Channel> channels)
    : two_js_(std::move(two_js)), triple_signs_(std::move(triple_signs)) {
  if (two_js_.size() < 3) {
    throw std::invalid_argument("BTKernel requires valence at least three.");
  }
  const bool has_half_integer_spin =
      std::any_of(two_js_.begin(), two_js_.end(), [](const Spin label) { return label % 2 != 0; });
  if (two_js_.size() > 4 && has_half_integer_spin) {
    throw std::invalid_argument(
        "Higher-valent half-integer sectors are not enabled yet. "
        "The generic Eq. 47 phase convention is validated for integer sectors; "
        "4-valent half-integer sectors use the validated closure reduction.");
  }
  if (channels.empty()) {
    channels_ = enumerate_left_comb_channels(two_js_);
  } else {
    channels_ = std::move(channels);
  }
  index_ = build_channel_index(channels_);
  for (const auto& sign : triple_signs_) {
    if (!(0 <= sign.i && sign.i < sign.j && sign.j < sign.k &&
          sign.k < static_cast<int>(two_js_.size()))) {
      throw std::invalid_argument("Triple signs must use zero-based ordered triples i<j<k.");
    }
    if (two_js_.size() == 4) {
      four_valent_sigma_ += sign.sign * four_valent_closure_sign(sign);
    }
  }
  if (two_js_.size() == 4) {
    four_valent_fast_path_ = true;
    for (std::size_t idx = 0; idx < channels_.size(); ++idx) {
      if (channels_[idx].size() != 1) {
        four_valent_fast_path_ = false;
        break;
      }
      if (idx > 0 && channels_[idx][0] <= channels_[idx - 1][0]) {
        four_valent_fast_path_ = false;
        break;
      }
    }
  }
  if (four_valent_fast_path_ && channels_.size() > 1) {
    four_valent_links_.assign(channels_.size() - 1, 0.0);
    for (std::size_t idx = 1; idx < channels_.size(); ++idx) {
      const Spin lower = channels_[idx - 1][0];
      const Spin upper = channels_[idx][0];
      if (upper == lower + 2) {
        four_valent_links_[idx - 1] =
            four_valent_offdiag_raw(two_js_[0], two_js_[1], two_js_[2], two_js_[3], upper);
      }
    }
  }
}

Spin BTKernel::full_a(const Channel& channel, const int paper_n) const {
  const int n = valence();
  if (paper_n == 1) {
    return two_js_[0];
  }
  if (2 <= paper_n && paper_n <= n - 2) {
    return channel[static_cast<std::size_t>(paper_n - 2)];
  }
  if (paper_n == n - 1) {
    return two_js_.back();
  }
  if (paper_n == n) {
    return 0;
  }
  throw std::out_of_range("paper recoupling label is out of range.");
}

bool BTKernel::same_outside(const Channel& bra,
                            const Channel& ket,
                            const int left_begin,
                            const int left_end,
                            const int right_begin,
                            const int right_end) const {
  for (int n = left_begin; n < left_end; ++n) {
    if (full_a(bra, n) != full_a(ket, n)) {
      return false;
    }
  }
  for (int n = right_begin; n < right_end; ++n) {
    if (full_a(bra, n) != full_a(ket, n)) {
      return false;
    }
  }
  return true;
}

double BTKernel::q123_matrix_element(const Channel& bra, const Channel& ket) const {
  if (bra.size() != ket.size()) {
    return 0.0;
  }
  for (std::size_t idx = 1; idx < bra.size(); ++idx) {
    if (bra[idx] != ket[idx]) {
      return 0.0;
    }
  }
  const Spin a2 = bra.empty() ? 0 : bra[0];
  const Spin a2p = ket.empty() ? 0 : ket[0];
  if (a2 == a2p + 2) {
    const Spin a3 = (valence() == 4) ? two_js_[3] : bra[1];
    return four_valent_offdiag_raw(two_js_[0], two_js_[1], two_js_[2], a3, a2);
  }
  if (a2p == a2 + 2) {
    const Spin a3 = (valence() == 4) ? two_js_[3] : ket[1];
    return -four_valent_offdiag_raw(two_js_[0], two_js_[1], two_js_[2], a3, a2p);
  }
  return 0.0;
}

double BTKernel::case_i1_j2(const Channel& bra,
                            const Channel& ket,
                            const int K,
                            Wigner6jCache& sixj) const {
  const int N = valence();
  if (!same_outside(bra, ket, 0, 0, K, N + 1)) {
    return 0.0;
  }

  const Spin a2 = full_a(bra, 2);
  const Spin a2p = full_a(ket, 2);
  const Spin aK = full_a(bra, K);
  const Spin aKm1 = full_a(bra, K - 1);
  const Spin aKm1p = full_a(ket, K - 1);

  int phase_twice = two_js_[K - 1] - two_js_[0] - two_js_[1] + aK + 2;
  for (int p = 3; p < K; ++p) {
    phase_twice -= two_js_[p - 1];
  }

  double value = 0.25 * phase_from_twice(phase_twice);
  value *= std::sqrt(x_factor(two_js_[1], two_js_[K - 1]));
  value *= sqrt_dim(a2, a2p);
  value *= sixj.get(two_js_[0], two_js_[1], a2, 2, a2p, two_js_[1]);
  for (int n = 3; n < K; ++n) {
    const Spin an = full_a(bra, n);
    const Spin anp = full_a(ket, n);
    const Spin anm1 = full_a(bra, n - 1);
    const Spin anm1p = full_a(ket, n - 1);
    value *= sqrt_dim(an, anp);
    value *= phase_from_twice(anm1p + anm1 + 2);
    value *= sixj.get(two_js_[n - 1], anm1p, anp, 2, an, anm1);
  }
  value *= sixj.get(aK, two_js_[K - 1], aKm1, 2, aKm1p, two_js_[K - 1]);
  value *= casimir(a2) - casimir(a2p);

  // closure phase and normalization for K > 3, checked against the magnetic basis
  return -2.0 * value;
}

double BTKernel::bracket_j(const Channel& bra,
                           const Channel& ket,
                           const int J,
                           Wigner6jCache& sixj) const {
  const Spin aJ = full_a(bra, J);
  const Spin aJp = full_a(ket, J);
  const Spin aJm1 = full_a(bra, J - 1);
  const Spin aJm1p = full_a(ket, J - 1);
  const Spin jJ = two_js_[J - 1];

  double term1 = phase_from_twice(aJp + aJm1p);
  term1 *= sixj.get(aJ, jJ, aJm1p, 2, aJm1, jJ);
  term1 *= sixj.get(aJm1p, jJ, aJp, 2, aJ, jJ);

  double term2 = phase_from_twice(aJ + aJm1);
  term2 *= sixj.get(aJp, jJ, aJm1, 2, aJm1p, jJ);
  term2 *= sixj.get(aJm1, jJ, aJ, 2, aJp, jJ);
  return term1 - term2;
}

double BTKernel::case_i1(const Channel& bra,
                         const Channel& ket,
                         const int J,
                         const int K,
                         Wigner6jCache& sixj) const {
  const int N = valence();
  if (!same_outside(bra, ket, 0, 0, K, N + 1)) {
    return 0.0;
  }

  const Spin aK = full_a(bra, K);
  const Spin aKm1 = full_a(bra, K - 1);
  const Spin aKm1p = full_a(ket, K - 1);
  const Spin aJ = full_a(bra, J);
  const Spin aJp = full_a(ket, J);

  int phase_twice = two_js_[K - 1] - two_js_[0] + aK + 2;
  for (int n = 2; n < J; ++n) {
    phase_twice += two_js_[n - 1];
  }
  for (int p = J + 1; p < K; ++p) {
    phase_twice -= two_js_[p - 1];
  }

  double value = 0.25 * phase_from_twice(phase_twice);
  value *= std::sqrt(x_factor(two_js_[0], two_js_[J - 1]));
  value *= std::sqrt(x_factor(two_js_[J - 1], two_js_[K - 1]));
  value *= sqrt_dim(aJ, aJp);
  for (int n = 2; n < J; ++n) {
    const Spin an = full_a(bra, n);
    const Spin anp = full_a(ket, n);
    const Spin anm1 = full_a(bra, n - 1);
    const Spin anm1p = full_a(ket, n - 1);
    value *= sqrt_dim(an, anp);
    value *= phase_from_twice(anm1p + anm1 + 2);
    value *= sixj.get(two_js_[n - 1], anm1p, anp, 2, an, anm1);
  }
  for (int n = J + 1; n < K; ++n) {
    const Spin an = full_a(bra, n);
    const Spin anp = full_a(ket, n);
    const Spin anm1 = full_a(bra, n - 1);
    const Spin anm1p = full_a(ket, n - 1);
    value *= sqrt_dim(an, anp);
    value *= phase_from_twice(anm1p + anm1 + 2);
    value *= sixj.get(two_js_[n - 1], anm1p, anp, 2, an, anm1);
  }
  value *= sixj.get(aK, two_js_[K - 1], aKm1, 2, aKm1p, two_js_[K - 1]);
  value *= bracket_j(bra, ket, J, sixj);
  return value;
}

double BTKernel::case_generic(const Channel& bra,
                              const Channel& ket,
                              const int I,
                              const int J,
                              const int K,
                              Wigner6jCache& sixj) const {
  const int N = valence();
  if (!same_outside(bra, ket, 2, I, K, N + 1)) {
    return 0.0;
  }

  const Spin aIm1 = full_a(bra, I - 1);
  const Spin aI = full_a(bra, I);
  const Spin aIp = full_a(ket, I);
  const Spin aJ = full_a(bra, J);
  const Spin aJp = full_a(ket, J);
  const Spin aK = full_a(bra, K);
  const Spin aKm1 = full_a(bra, K - 1);
  const Spin aKm1p = full_a(ket, K - 1);

  int phase_twice = two_js_[K - 1] + two_js_[I - 1] + aIm1 + aK;
  phase_twice += aI - aIp;
  for (int n = I + 1; n < J; ++n) {
    phase_twice += two_js_[n - 1];
  }
  for (int p = J + 1; p < K; ++p) {
    phase_twice -= two_js_[p - 1];
  }

  double value = 0.25 * phase_from_twice(phase_twice);
  value *= std::sqrt(x_factor(two_js_[I - 1], two_js_[J - 1]));
  value *= std::sqrt(x_factor(two_js_[J - 1], two_js_[K - 1]));
  value *= sqrt_dim(aI, aIp);
  value *= sqrt_dim(aJ, aJp);
  value *= sixj.get(aIm1, two_js_[I - 1], aI, 2, aIp, two_js_[I - 1]);
  for (int n = I + 1; n < J; ++n) {
    const Spin an = full_a(bra, n);
    const Spin anp = full_a(ket, n);
    const Spin anm1 = full_a(bra, n - 1);
    const Spin anm1p = full_a(ket, n - 1);
    value *= sqrt_dim(an, anp);
    value *= phase_from_twice(anm1p + anm1 + 2);
    value *= sixj.get(two_js_[n - 1], anm1p, anp, 2, an, anm1);
  }
  for (int n = J + 1; n < K; ++n) {
    const Spin an = full_a(bra, n);
    const Spin anp = full_a(ket, n);
    const Spin anm1 = full_a(bra, n - 1);
    const Spin anm1p = full_a(ket, n - 1);
    value *= sqrt_dim(an, anp);
    value *= phase_from_twice(anm1p + anm1 + 2);
    value *= sixj.get(two_js_[n - 1], anm1p, anp, 2, an, anm1);
  }
  value *= sixj.get(aK, two_js_[K - 1], aKm1, 2, aKm1p, two_js_[K - 1]);
  value *= bracket_j(bra, ket, J, sixj);
  return value;
}

double BTKernel::matrix_element_cached(const Channel& bra,
                                       const Channel& ket,
                                       const int i,
                                       const int j,
                                       const int k,
                                       Wigner6jCache& sixj) const {
  const int I = i + 1;
  const int J = j + 1;
  const int K = k + 1;
  if (!(1 <= I && I < J && J < K && K <= valence())) {
    throw std::invalid_argument("Invalid zero-based triple for matrix_element.");
  }
  if (four_valent_fast_path_) {
    const double base = q123_matrix_element(bra, ket);
    if (I == 1 && J == 2 && K == 3) {
      return base;
    }
    if (I == 1 && J == 2 && K == 4) {
      return -base;
    }
    if (I == 1 && J == 3 && K == 4) {
      return base;
    }
    if (I == 2 && J == 3 && K == 4) {
      return -base;
    }
  }
  if (I == 1 && J == 2 && K == 3) {
    return q123_matrix_element(bra, ket);
  }
  if (I == 1 && J == 2) {
    return case_i1_j2(bra, ket, K, sixj);
  }
  if (I == 1) {
    return case_i1(bra, ket, J, K, sixj);
  }
  return case_generic(bra, ket, I, J, K, sixj);
}

double BTKernel::matrix_element(const Channel& bra, const Channel& ket, const int i, const int j, const int k) const {
  return matrix_element_cached(bra, ket, i, j, k, sixj_);
}

std::vector<Channel> BTKernel::candidate_bras(const Channel& ket, const int i, const int j, const int k) const {
  const int I = i + 1;
  const int K = k + 1;
  const int N = valence();

  if (N == 4 || (I == 1 && j + 1 == 2 && K == 3)) {
    std::vector<Channel> out;
    for (const int delta : {-2, 2}) {
      Channel candidate = ket;
      if (candidate.empty()) {
        continue;
      }
      candidate[0] += delta;
      if (index_.find(candidate) != index_.end()) {
        out.push_back(std::move(candidate));
      }
    }
    return out;
  }

  const int begin_n = (I == 1) ? 2 : I;
  const int end_n = std::min(K - 1, N - 2);
  std::vector<Channel> out;
  Channel candidate = ket;

  std::function<void(int)> rec = [&](const int n) {
    if (n > end_n) {
      if (index_.find(candidate) != index_.end()) {
        out.push_back(candidate);
      }
      return;
    }
    const std::size_t slot = static_cast<std::size_t>(n - 2);
    const Spin original = ket[slot];
    for (const int delta : {-2, 0, 2}) {
      candidate[slot] = original + delta;
      rec(n + 1);
    }
    candidate[slot] = original;
  };

  if (begin_n <= end_n) {
    rec(begin_n);
  }
  return out;
}

void BTKernel::scatter_column(const Channel& ket,
                              const std::complex<double> amplitude,
                              std::complex<double>* output,
                              Wigner6jCache& sixj) const {
  if (std::abs(amplitude) <= kDropTol) {
    return;
  }

  const std::complex<double> imag{0.0, 1.0};
  const int n = valence();

  if (n == 4) {
    if (ket.empty() || four_valent_sigma_ == 0) {
      return;
    }

    auto scatter_four_valent = [&](const Spin bra_a2, const double q_comm) {
      if (std::abs(q_comm) <= kDropTol) {
        return;
      }
      Channel bra{bra_a2};
      const auto row_it = index_.find(bra);
      if (row_it == index_.end()) {
        return;
      }
      output[row_it->second] += imag * static_cast<double>(four_valent_sigma_) * q_comm * amplitude;
    };

    const Spin ket_a2 = ket[0];
    scatter_four_valent(ket_a2 - 2,
                        -four_valent_offdiag_raw(two_js_[0], two_js_[1], two_js_[2], two_js_[3], ket_a2));
    scatter_four_valent(ket_a2 + 2,
                        four_valent_offdiag_raw(two_js_[0], two_js_[1], two_js_[2], two_js_[3], ket_a2 + 2));
    return;
  }

  for (const auto& triple : triple_signs_) {
    if (triple.sign == 0) {
      continue;
    }

    const int I = triple.i + 1;
    const int J = triple.j + 1;
    const int K = triple.k + 1;

    auto scatter = [&](const Channel& bra) {
      const auto row_it = index_.find(bra);
      if (row_it == index_.end()) {
        return;
      }
      const double q_comm = matrix_element_cached(bra, ket, triple.i, triple.j, triple.k, sixj);
      if (std::abs(q_comm) <= kDropTol) {
        return;
      }
      output[row_it->second] += imag * static_cast<double>(triple.sign) * q_comm * amplitude;
    };

    if (n == 4 || (I == 1 && J == 2 && K == 3)) {
      if (ket.empty()) {
        continue;
      }
      Channel candidate = ket;
      candidate[0] = ket[0] - 2;
      scatter(candidate);
      candidate[0] = ket[0] + 2;
      scatter(candidate);
      continue;
    }

    const int begin_n = (I == 1) ? 2 : I;
    const int end_n = std::min(K - 1, n - 2);
    if (begin_n > end_n) {
      continue;
    }

    Channel candidate = ket;
    auto rec = [&](auto&& self, const int current_n) -> void {
      if (current_n > end_n) {
        scatter(candidate);
        return;
      }
      const std::size_t slot = static_cast<std::size_t>(current_n - 2);
      const Spin original = ket[slot];
      candidate[slot] = original - 2;
      self(self, current_n + 1);
      candidate[slot] = original;
      self(self, current_n + 1);
      candidate[slot] = original + 2;
      self(self, current_n + 1);
      candidate[slot] = original;
    };
    rec(rec, begin_n);
  }
}

void BTKernel::accumulate_abs_column_bound(const Channel& ket,
                                           double* row_sums,
                                           Wigner6jCache& sixj) const {
  const int n = valence();

  if (n == 4) {
    if (ket.empty() || four_valent_sigma_ == 0) {
      return;
    }

    auto add_four_valent = [&](const Spin bra_a2, const double q_comm) {
      if (std::abs(q_comm) <= kDropTol) {
        return;
      }
      Channel bra{bra_a2};
      const auto row_it = index_.find(bra);
      if (row_it == index_.end()) {
        return;
      }
      row_sums[row_it->second] += std::abs(static_cast<double>(four_valent_sigma_) * q_comm);
    };

    const Spin ket_a2 = ket[0];
    add_four_valent(ket_a2 - 2,
                    -four_valent_offdiag_raw(two_js_[0], two_js_[1], two_js_[2], two_js_[3], ket_a2));
    add_four_valent(ket_a2 + 2,
                    four_valent_offdiag_raw(two_js_[0], two_js_[1], two_js_[2], two_js_[3], ket_a2 + 2));
    return;
  }

  for (const auto& triple : triple_signs_) {
    if (triple.sign == 0) {
      continue;
    }

    const int I = triple.i + 1;
    const int J = triple.j + 1;
    const int K = triple.k + 1;

    auto accumulate = [&](const Channel& bra) {
      const auto row_it = index_.find(bra);
      if (row_it == index_.end()) {
        return;
      }
      const double q_comm = matrix_element_cached(bra, ket, triple.i, triple.j, triple.k, sixj);
      if (std::abs(q_comm) <= kDropTol) {
        return;
      }

      // bound each triple seperately so cancellations cannot lower the estimate
      row_sums[row_it->second] += std::abs(static_cast<double>(triple.sign) * q_comm);
    };

    if (n == 4 || (I == 1 && J == 2 && K == 3)) {
      if (ket.empty()) {
        continue;
      }
      Channel candidate = ket;
      candidate[0] = ket[0] - 2;
      accumulate(candidate);
      candidate[0] = ket[0] + 2;
      accumulate(candidate);
      continue;
    }

    const int begin_n = (I == 1) ? 2 : I;
    const int end_n = std::min(K - 1, n - 2);
    if (begin_n > end_n) {
      continue;
    }

    Channel candidate = ket;
    auto rec = [&](auto&& self, const int current_n) -> void {
      if (current_n > end_n) {
        accumulate(candidate);
        return;
      }
      const std::size_t slot = static_cast<std::size_t>(current_n - 2);
      const Spin original = ket[slot];
      candidate[slot] = original - 2;
      self(self, current_n + 1);
      candidate[slot] = original;
      self(self, current_n + 1);
      candidate[slot] = original + 2;
      self(self, current_n + 1);
      candidate[slot] = original;
    };
    rec(rec, begin_n);
  }
}

double BTKernel::row_sum_bound() const {
  if (dim() == 0) {
    return 0.0;
  }
  if (four_valent_fast_path_) {
    const double sigma_abs = std::abs(static_cast<double>(four_valent_sigma_));
    if (sigma_abs == 0.0 || four_valent_links_.empty()) {
      return 0.0;
    }
    double bound = 0.0;
    for (std::size_t row = 0; row < dim(); ++row) {
      double row_sum = 0.0;
      if (row > 0) {
        row_sum += four_valent_links_[row - 1];
      }
      if (row + 1 < dim()) {
        row_sum += four_valent_links_[row];
      }
      bound = std::max(bound, sigma_abs * row_sum);
    }
    return bound;
  }

  const int threads = requested_openmp_threads(dim());
  if (threads <= 1) {
    std::vector<double> row_sums(dim(), 0.0);
    thread_local Wigner6jCache thread_sixj;
    for (const auto& ket : channels_) {
      accumulate_abs_column_bound(ket, row_sums.data(), thread_sixj);
    }
    return *std::max_element(row_sums.begin(), row_sums.end());
  }

#if defined(BTKERNEL_HAS_OPENMP)
  std::vector<std::vector<double>> partials(static_cast<std::size_t>(threads),
                                            std::vector<double>(dim(), 0.0));

#pragma omp parallel num_threads(threads)
  {
    const int tid = omp_get_thread_num();
    thread_local Wigner6jCache local_sixj;
    auto* local_rows = partials[static_cast<std::size_t>(tid)].data();

#pragma omp for schedule(dynamic, 32)
    for (std::ptrdiff_t col = 0; col < static_cast<std::ptrdiff_t>(dim()); ++col) {
      accumulate_abs_column_bound(channels_[static_cast<std::size_t>(col)],
                                  local_rows,
                                  local_sixj);
    }
  }

  double bound = 0.0;
#pragma omp parallel for schedule(static) reduction(max : bound) num_threads(threads)
  for (std::ptrdiff_t row = 0; row < static_cast<std::ptrdiff_t>(dim()); ++row) {
    double row_sum = 0.0;
    for (int tid = 0; tid < threads; ++tid) {
      row_sum += partials[static_cast<std::size_t>(tid)][static_cast<std::size_t>(row)];
    }
    bound = std::max(bound, row_sum);
  }
  return bound;
#else
  (void) threads;
  std::vector<double> row_sums(dim(), 0.0);
  thread_local Wigner6jCache thread_sixj;
  for (const auto& ket : channels_) {
    accumulate_abs_column_bound(ket, row_sums.data(), thread_sixj);
  }
  return *std::max_element(row_sums.begin(), row_sums.end());
#endif
}

void BTKernel::apply_range_into(const std::complex<double>* input,
                                std::complex<double>* output,
                                const std::size_t begin,
                                const std::size_t end,
                                Wigner6jCache& sixj) const {
  for (std::size_t col = begin; col < end; ++col) {
    scatter_column(channels_[col], input[col], output, sixj);
  }
}

void BTKernel::apply_range_into_parallel(const std::complex<double>* input,
                                         std::complex<double>* output,
                                         const std::size_t begin,
                                         const std::size_t end) const {
  const std::size_t work_items = end - begin;
  const int threads = requested_openmp_threads(work_items);
  if (threads <= 1) {
    thread_local Wigner6jCache thread_sixj;
    apply_range_into(input, output, begin, end, thread_sixj);
    return;
  }

#if defined(BTKERNEL_HAS_OPENMP)
  std::vector<ComplexVector> partials(static_cast<std::size_t>(threads), ComplexVector(dim(), {0.0, 0.0}));

#pragma omp parallel num_threads(threads)
  {
    const int tid = omp_get_thread_num();
    thread_local Wigner6jCache local_sixj;
    auto* local_output = partials[static_cast<std::size_t>(tid)].data();

#pragma omp for schedule(dynamic, 32)
    for (std::ptrdiff_t col = static_cast<std::ptrdiff_t>(begin);
         col < static_cast<std::ptrdiff_t>(end);
         ++col) {
      scatter_column(channels_[static_cast<std::size_t>(col)],
                     input[static_cast<std::size_t>(col)],
                     local_output,
                     local_sixj);
    }
  }

#pragma omp parallel for schedule(static) num_threads(threads)
  for (std::ptrdiff_t row = 0; row < static_cast<std::ptrdiff_t>(dim()); ++row) {
    std::complex<double> sum{0.0, 0.0};
    for (int tid = 0; tid < threads; ++tid) {
      sum += partials[static_cast<std::size_t>(tid)][static_cast<std::size_t>(row)];
    }
    output[static_cast<std::size_t>(row)] += sum;
  }
#else
  (void) threads;
  thread_local Wigner6jCache thread_sixj;
  apply_range_into(input, output, begin, end, thread_sixj);
#endif
}

void BTKernel::apply_four_valent_into(const std::complex<double>* input,
                                      std::complex<double>* output) const {
  const std::size_t d = dim();
  std::fill(output, output + d, std::complex<double>{0.0, 0.0});
  if (d == 0 || four_valent_sigma_ == 0 || four_valent_links_.empty()) {
    return;
  }

  const std::complex<double> imag_sigma{0.0, static_cast<double>(four_valent_sigma_)};

#if defined(BTKERNEL_HAS_OPENMP)
#pragma omp parallel for schedule(static) if (d >= 8192)
#endif
  for (std::ptrdiff_t row_signed = 0; row_signed < static_cast<std::ptrdiff_t>(d); ++row_signed) {
    const std::size_t row = static_cast<std::size_t>(row_signed);
    std::complex<double> value{0.0, 0.0};
    if (row > 0) {
      value += four_valent_links_[row - 1] * input[row - 1];
    }
    if (row + 1 < d) {
      value -= four_valent_links_[row] * input[row + 1];
    }
    output[row] = imag_sigma * value;
  }
}

void BTKernel::apply_into(const std::complex<double>* input,
                          std::complex<double>* output,
                          const std::size_t size) const {
  if (size != dim()) {
    throw std::invalid_argument("BTKernel::apply_into input dimension mismatch.");
  }
  if (input == nullptr || output == nullptr) {
    throw std::invalid_argument("BTKernel::apply_into received a null buffer.");
  }

  // keep the recoupling traversal used for the publication error check
  std::fill(output, output + dim(), std::complex<double>{0.0, 0.0});
  apply_range_into_parallel(input, output, 0, dim());
}

void BTKernel::apply_into_mpi_world(const std::complex<double>* input,
                                    std::complex<double>* output,
                                    const std::size_t size) const {
  if (size != dim()) {
    throw std::invalid_argument("BTKernel::apply_into_mpi_world input dimension mismatch.");
  }
  if (input == nullptr || output == nullptr) {
    throw std::invalid_argument("BTKernel::apply_into_mpi_world received a null buffer.");
  }

#if defined(BTKERNEL_HAS_MPI)
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    throw std::runtime_error("BTKernel::apply_into_mpi_world requires MPI_Init before use.");
  }

  int rank = 0;
  int ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);

  const std::size_t begin = (dim() * static_cast<std::size_t>(rank)) / static_cast<std::size_t>(ranks);
  const std::size_t end = (dim() * static_cast<std::size_t>(rank + 1)) / static_cast<std::size_t>(ranks);

  ComplexVector local(dim(), {0.0, 0.0});
  apply_range_into_parallel(input, local.data(), begin, end);
  mpi_allreduce_complex_sum(local.data(), output, dim());
#else
  (void) input;
  (void) output;
  throw std::runtime_error("BTKernel was built without MPI support.");
#endif
}

void BTKernel::apply_into(const ComplexVector& input, ComplexVector& output) const {
  if (input.size() != dim()) {
    throw std::invalid_argument("BTKernel::apply_into input dimension mismatch.");
  }
  output.resize(dim());
  apply_into(input.data(), output.data(), input.size());
}

ComplexVector BTKernel::apply(const ComplexVector& input) const {
  ComplexVector out(dim(), std::complex<double>{0.0, 0.0});
  apply_into(input, out);
  return out;
}

bool openmp_enabled() noexcept {
#if defined(BTKERNEL_HAS_OPENMP)
  return true;
#else
  return false;
#endif
}

int openmp_max_threads() noexcept {
#if defined(BTKERNEL_HAS_OPENMP)
  return omp_get_max_threads();
#else
  return 1;
#endif
}

void openmp_set_num_threads(const int threads) {
  if (threads < 1) {
    throw std::invalid_argument("OpenMP thread count must be positive.");
  }
#if defined(BTKERNEL_HAS_OPENMP)
  omp_set_num_threads(threads);
#else
  if (threads != 1) {
    throw std::runtime_error("BTKernel was built without OpenMP support.");
  }
#endif
}

bool mpi_enabled() noexcept {
#if defined(BTKERNEL_HAS_MPI)
  return true;
#else
  return false;
#endif
}

bool mpi_initialized() noexcept {
#if defined(BTKERNEL_HAS_MPI)
  int initialized = 0;
  MPI_Initialized(&initialized);
  return initialized != 0;
#else
  return false;
#endif
}

bool mpi_finalized() noexcept {
#if defined(BTKERNEL_HAS_MPI)
  int finalized = 0;
  MPI_Finalized(&finalized);
  return finalized != 0;
#else
  return false;
#endif
}

int mpi_init_thread() {
#if defined(BTKERNEL_HAS_MPI)
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (finalized) {
    throw std::runtime_error("MPI has already been finalized.");
  }
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized) {
    return 0;
  }
  int argc = 0;
  char** argv = nullptr;
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  return provided;
#else
  throw std::runtime_error("BTKernel was built without MPI support.");
#endif
}

void mpi_finalize() {
#if defined(BTKERNEL_HAS_MPI)
  int initialized = 0;
  int finalized = 0;
  MPI_Initialized(&initialized);
  MPI_Finalized(&finalized);
  if (initialized && !finalized) {
    MPI_Finalize();
  }
#else
  throw std::runtime_error("BTKernel was built without MPI support.");
#endif
}

int mpi_world_rank() {
#if defined(BTKERNEL_HAS_MPI)
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    return 0;
  }
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
#else
  return 0;
#endif
}

int mpi_world_size() {
#if defined(BTKERNEL_HAS_MPI)
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    return 1;
  }
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size;
#else
  return 1;
#endif
}

std::vector<Spin> parse_spin_list(const std::string& text) {
  std::vector<Spin> spins;
  for (const auto& part : split(text, ',')) {
    spins.push_back(std::stoi(part));
  }
  if (spins.empty()) {
    throw std::invalid_argument("empty spin list");
  }
  return spins;
}

std::vector<TripleSign> parse_triple_signs(const std::string& text, const int valence) {
  std::vector<TripleSign> signs;
  if (text == "none" || text == "NONE" || text == "None") {
    return signs;
  }
  if (text == "all" || text.empty()) {
    for (int i = 0; i < valence; ++i) {
      for (int j = i + 1; j < valence; ++j) {
        for (int k = j + 1; k < valence; ++k) {
          signs.push_back({i, j, k, 1});
        }
      }
    }
    return signs;
  }
  for (const auto& entry : split(text, ';')) {
    const auto pieces = split(entry, ':');
    if (pieces.size() != 2) {
      throw std::invalid_argument("triple signs must have form i,j,k:sign;...");
    }
    const auto ijk = split(pieces[0], ',');
    if (ijk.size() != 3) {
      throw std::invalid_argument("triple must have three comma-separated indices");
    }
    signs.push_back({std::stoi(ijk[0]), std::stoi(ijk[1]), std::stoi(ijk[2]), std::stoi(pieces[1])});
  }
  return signs;
}

}
