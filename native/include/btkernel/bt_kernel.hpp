#pragma once

#include "btkernel/left_comb.hpp"
#include "btkernel/wigner6j.hpp"

#include <complex>

namespace btkernel {

// q_IJK = [(J_I + J_J)^2, (J_J + J_K)^2] and Q_v = i sum eps_IJK q_IJK
class BTKernel {
 public:

  BTKernel(std::vector<Spin> two_js,
           std::vector<TripleSign> triple_signs,
           std::vector<Channel> channels = {});

  int valence() const noexcept { return static_cast<int>(two_js_.size()); }

  std::size_t dim() const noexcept { return channels_.size(); }

  const std::vector<Spin>& two_js() const noexcept { return two_js_; }

  const std::vector<Channel>& channels() const noexcept { return channels_; }

  std::size_t sixj_cache_size() const noexcept { return sixj_.size(); }

  int four_valent_sigma() const noexcept { return four_valent_sigma_; }

  const std::vector<double>& four_valent_links() const noexcept { return four_valent_links_; }

  bool four_valent_fast_path() const noexcept { return four_valent_fast_path_; }

  double row_sum_bound() const;

  ComplexVector apply(const ComplexVector& input) const;

  // input and output must not overlap
  void apply_into(const std::complex<double>* input,
                  std::complex<double>* output,
                  std::size_t size) const;

  void apply_into(const ComplexVector& input, ComplexVector& output) const;

  // every rank supplies the full input and participates in the reduction
  void apply_into_mpi_world(const std::complex<double>* input,
                            std::complex<double>* output,
                            std::size_t size) const;

  double matrix_element(const Channel& bra, const Channel& ket, int i, int j, int k) const;

 private:
  std::vector<Spin> two_js_;
  std::vector<TripleSign> triple_signs_;
  std::vector<Channel> channels_;
  ChannelIndex index_;
  mutable Wigner6jCache sixj_;
  int four_valent_sigma_ = 0;
  std::vector<double> four_valent_links_;
  bool four_valent_fast_path_ = false;

  Spin full_a(const Channel& channel, int paper_n) const;
  double q123_matrix_element(const Channel& bra, const Channel& ket) const;
  double case_i1_j2(const Channel& bra, const Channel& ket, int K, Wigner6jCache& sixj) const;
  double case_i1(const Channel& bra, const Channel& ket, int J, int K, Wigner6jCache& sixj) const;
  double case_generic(const Channel& bra, const Channel& ket, int I, int J, int K, Wigner6jCache& sixj) const;
  double bracket_j(const Channel& bra, const Channel& ket, int J, Wigner6jCache& sixj) const;
  double matrix_element_cached(const Channel& bra,
                               const Channel& ket,
                               int i,
                               int j,
                               int k,
                               Wigner6jCache& sixj) const;

  bool same_outside(const Channel& bra,
                    const Channel& ket,
                    int left_begin,
                    int left_end,
                    int right_begin,
                    int right_end) const;

  std::vector<Channel> candidate_bras(const Channel& ket, int i, int j, int k) const;
  void accumulate_abs_column_bound(const Channel& ket,
                                   double* row_sums,
                                   Wigner6jCache& sixj) const;
  void scatter_column(const Channel& ket,
                      std::complex<double> amplitude,
                      std::complex<double>* output,
                      Wigner6jCache& sixj) const;
  void apply_range_into(const std::complex<double>* input,
                        std::complex<double>* output,
                        std::size_t begin,
                        std::size_t end,
                        Wigner6jCache& sixj) const;
  void apply_range_into_parallel(const std::complex<double>* input,
                                 std::complex<double>* output,
                                 std::size_t begin,
                                 std::size_t end) const;
  void apply_four_valent_into(const std::complex<double>* input,
                              std::complex<double>* output) const;
};

double four_valent_offdiag(const std::vector<Spin>& two_js4, Spin upper_channel);

bool openmp_enabled() noexcept;
int openmp_max_threads() noexcept;
void openmp_set_num_threads(int threads);
bool mpi_enabled() noexcept;
bool mpi_initialized() noexcept;
bool mpi_finalized() noexcept;
int mpi_init_thread();
void mpi_finalize();
int mpi_world_rank();
int mpi_world_size();

std::vector<Spin> parse_spin_list(const std::string& text);
std::vector<TripleSign> parse_triple_signs(const std::string& text, int valence);

}
