#pragma once

#include "btkernel/bt_kernel.hpp"

#include <cstdint>

namespace btkernel {

struct SRQGrid {
  int shifts = 0;
  int kminus = 0;
  int kplus = 0;
  double h = 0.0;
  std::vector<double> taus;
  std::vector<double> weights;
};

struct SRQDiagnostics {
  double max_relres = 0.0;
  double mean_relres = 0.0;
  int shifted_solves = 0;
  std::int64_t cg_iterations = 0;
  bool converged = true;
};

SRQGrid make_srq_grid(int shifts);

class VolumeSRQ {
 public:

  VolumeSRQ(std::vector<Spin> two_js,
            std::vector<TripleSign> triple_signs,
            // a negative value requests the streamed sector row-sum bound
            double lambda_bound,
            int shifts,
            double cg_tol = 1.0e-12,
            int cg_maxiter = 0,
            std::vector<Channel> channels = {});

  std::size_t dim() const noexcept { return kernel_.dim(); }
  int valence() const noexcept { return kernel_.valence(); }
  const std::vector<Spin>& two_js() const noexcept { return kernel_.two_js(); }
  const std::vector<Channel>& channels() const noexcept { return kernel_.channels(); }
  double lambda_bound() const noexcept { return lambda_bound_; }
  int shifts() const noexcept { return grid_.shifts; }
  double cg_tol() const noexcept { return cg_tol_; }
  int cg_maxiter() const noexcept { return cg_maxiter_; }
  const SRQGrid& grid() const noexcept { return grid_; }
  const SRQDiagnostics& last_diagnostics() const noexcept { return last_diagnostics_; }

  ComplexVector apply(const ComplexVector& input);

  void apply_into(const std::complex<double>* input,
                  std::complex<double>* output,
                  std::size_t size);

  void apply_into_mpi_world(const std::complex<double>* input,
                            std::complex<double>* output,
                            std::size_t size);

  void apply_abar_into(const std::complex<double>* input,
                       std::complex<double>* output,
                       std::size_t size);

 private:
  struct WorkBuffers {
    ComplexVector q_tmp;
    ComplexVector qq_tmp;
    ComplexVector rhs;
    ComplexVector x;
    ComplexVector r;
    ComplexVector p;
    ComplexVector ap;
    ComplexVector shifted_ap;
    std::vector<double> lanczos_alpha;
    std::vector<double> lanczos_beta;
    std::vector<double> tridiagonal_cprime;
    std::vector<double> tridiagonal_solution;
    std::vector<double> combined_coefficients;
    std::vector<double> relres_shift;
    ComplexVector banded_rhs;
    ComplexVector banded_acc;
    ComplexVector banded_dprime;
    std::vector<double> banded_cprime;
  };

  struct SolveStats {
    double relres = 0.0;
    int iterations = 0;
    bool converged = true;
  };

  BTKernel kernel_;
  double lambda_bound_ = 0.0;
  double lambda_sq_ = 0.0;
  SRQGrid grid_;
  double cg_tol_ = 1.0e-12;
  int cg_maxiter_ = 0;
  SRQDiagnostics last_diagnostics_{};

  WorkBuffers work_;
  bool four_valent_banded_ = false;
  std::vector<double> four_abar_diag_;
  std::vector<double> four_abar_off2_;

  int default_maxiter() const;
  void validate_vector_size(std::size_t size, const char* caller) const;
  void ensure_work(WorkBuffers& work) const;
  void initialize_four_valent_bands();

  void apply_abar_with_work(const std::complex<double>* input,
                            std::complex<double>* output,
                            WorkBuffers& work,
                            bool mpi);
  void apply_four_valent_abar(const std::complex<double>* input,
                              std::complex<double>* output) const;
  void solve_four_valent_shift(double tau,
                               const std::complex<double>* rhs,
                               std::complex<double>* solution,
                               WorkBuffers& work) const;
  void apply_four_valent_srq(const std::complex<double>* input,
                             std::complex<double>* output,
                             WorkBuffers& work);

  SolveStats solve_shifted(double tau,
                           const std::complex<double>* rhs,
                           std::complex<double>* solution,
                           WorkBuffers& work,
                           bool mpi);

  SRQDiagnostics solve_shifted_sum_lanczos(const std::complex<double>* rhs,
                                           std::complex<double>* weighted_sum,
                                           WorkBuffers& work,
                                           bool mpi);

  void apply_impl(const std::complex<double>* input,
                  std::complex<double>* output,
                  bool mpi);
};

}
