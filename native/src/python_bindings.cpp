#include "btkernel/bt_kernel.hpp"
#include "btkernel/left_comb.hpp"
#include "btkernel/volume_srq.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {

using btkernel::BTKernel;
using btkernel::Channel;
using btkernel::ComplexVector;
using btkernel::Spin;
using btkernel::TripleSign;
using btkernel::VolumeSRQ;

using InputArray =
    nb::ndarray<nb::numpy, const std::complex<double>, nb::ndim<1>, nb::c_contig>;
using OutputArray =
    nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<1>, nb::c_contig>;

BTKernel make_kernel(std::vector<Spin> two_js,
                     const std::string& triple_signs,
                     std::vector<Channel> channels) {
  const int valence = static_cast<int>(two_js.size());
  auto signs = btkernel::parse_triple_signs(triple_signs, valence);
  return BTKernel(std::move(two_js), std::move(signs), std::move(channels));
}

VolumeSRQ make_volume_srq(std::vector<Spin> two_js,
                          const std::string& triple_signs,
                          std::vector<Channel> channels,
                          const double lambda_bound,
                          const int shifts,
                          const double cg_tol,
                          const int cg_maxiter) {
  const int valence = static_cast<int>(two_js.size());
  auto signs = btkernel::parse_triple_signs(triple_signs, valence);
  return VolumeSRQ(std::move(two_js),
                   std::move(signs),
                   lambda_bound,
                   shifts,
                   cg_tol,
                   cg_maxiter,
                   std::move(channels));
}

class PyApplyQ {
 public:
  PyApplyQ(std::vector<Spin> two_js,
           const std::string& triple_signs,
           std::vector<Channel> channels)
      : kernel_(make_kernel(std::move(two_js), triple_signs, std::move(channels))) {}

  const BTKernel& kernel() const noexcept { return kernel_; }

 private:
  BTKernel kernel_;
};

class PyVolumeSRQ {
 public:
  PyVolumeSRQ(std::vector<Spin> two_js,
              const std::string& triple_signs,
              std::vector<Channel> channels,
              double lambda_bound,
              int shifts,
              double cg_tol,
              int cg_maxiter)
      : volume_(make_volume_srq(std::move(two_js),
                                triple_signs,
                                std::move(channels),
                                lambda_bound,
                                shifts,
                                cg_tol,
                                cg_maxiter)) {}

  VolumeSRQ& volume() noexcept { return volume_; }
  const VolumeSRQ& volume() const noexcept { return volume_; }

 private:
  VolumeSRQ volume_;
};

OutputArray apply_array(const PyApplyQ& op, const InputArray& input) {
  const BTKernel& kernel = op.kernel();
  const std::size_t dim = kernel.dim();
  if (input.shape(0) != dim) {
    throw std::invalid_argument("ApplyQ input dimension mismatch.");
  }

  auto* data = new std::complex<double>[dim];
  nb::capsule owner(data, [](void* ptr) noexcept {
    delete[] static_cast<std::complex<double>*>(ptr);
  });

  {
    nb::gil_scoped_release release;
    kernel.apply_into(input.data(), data, dim);
  }

  return OutputArray(data, {dim}, owner);
}

void apply_into_array(const PyApplyQ& op, const InputArray& input, const OutputArray& output) {
  const BTKernel& kernel = op.kernel();
  const std::size_t dim = kernel.dim();
  if (input.shape(0) != dim || output.shape(0) != dim) {
    throw std::invalid_argument("ApplyQ input/output dimension mismatch.");
  }

  nb::gil_scoped_release release;
  kernel.apply_into(input.data(), output.data(), dim);
}

OutputArray apply_mpi_array(const PyApplyQ& op, const InputArray& input) {
  const BTKernel& kernel = op.kernel();
  const std::size_t dim = kernel.dim();
  if (input.shape(0) != dim) {
    throw std::invalid_argument("ApplyQ MPI input dimension mismatch.");
  }

  auto* data = new std::complex<double>[dim];
  nb::capsule owner(data, [](void* ptr) noexcept {
    delete[] static_cast<std::complex<double>*>(ptr);
  });

  {
    nb::gil_scoped_release release;
    kernel.apply_into_mpi_world(input.data(), data, dim);
  }

  return OutputArray(data, {dim}, owner);
}

void apply_mpi_into_array(const PyApplyQ& op, const InputArray& input, const OutputArray& output) {
  const BTKernel& kernel = op.kernel();
  const std::size_t dim = kernel.dim();
  if (input.shape(0) != dim || output.shape(0) != dim) {
    throw std::invalid_argument("ApplyQ MPI input/output dimension mismatch.");
  }

  nb::gil_scoped_release release;
  kernel.apply_into_mpi_world(input.data(), output.data(), dim);
}

ComplexVector apply_vector(const PyApplyQ& op, const ComplexVector& input) {
  return op.kernel().apply(input);
}

OutputArray srq_apply_array(PyVolumeSRQ& op, const InputArray& input) {
  const std::size_t dim = op.volume().dim();
  if (input.shape(0) != dim) {
    throw std::invalid_argument("VolumeSRQ input dimension mismatch.");
  }

  auto* data = new std::complex<double>[dim];
  nb::capsule owner(data, [](void* ptr) noexcept {
    delete[] static_cast<std::complex<double>*>(ptr);
  });

  {
    nb::gil_scoped_release release;
    op.volume().apply_into(input.data(), data, dim);
  }

  return OutputArray(data, {dim}, owner);
}

void srq_apply_into_array(PyVolumeSRQ& op, const InputArray& input, const OutputArray& output) {
  const std::size_t dim = op.volume().dim();
  if (input.shape(0) != dim || output.shape(0) != dim) {
    throw std::invalid_argument("VolumeSRQ input/output dimension mismatch.");
  }

  nb::gil_scoped_release release;
  op.volume().apply_into(input.data(), output.data(), dim);
}

OutputArray srq_apply_mpi_array(PyVolumeSRQ& op, const InputArray& input) {
  const std::size_t dim = op.volume().dim();
  if (input.shape(0) != dim) {
    throw std::invalid_argument("VolumeSRQ MPI input dimension mismatch.");
  }

  auto* data = new std::complex<double>[dim];
  nb::capsule owner(data, [](void* ptr) noexcept {
    delete[] static_cast<std::complex<double>*>(ptr);
  });

  {
    nb::gil_scoped_release release;
    op.volume().apply_into_mpi_world(input.data(), data, dim);
  }

  return OutputArray(data, {dim}, owner);
}

void srq_apply_mpi_into_array(PyVolumeSRQ& op, const InputArray& input, const OutputArray& output) {
  const std::size_t dim = op.volume().dim();
  if (input.shape(0) != dim || output.shape(0) != dim) {
    throw std::invalid_argument("VolumeSRQ MPI input/output dimension mismatch.");
  }

  nb::gil_scoped_release release;
  op.volume().apply_into_mpi_world(input.data(), output.data(), dim);
}

nb::dict srq_diagnostics_dict(const PyVolumeSRQ& op) {
  const auto& diagnostics = op.volume().last_diagnostics();
  nb::dict out;
  out["max_relres"] = diagnostics.max_relres;
  out["mean_relres"] = diagnostics.mean_relres;
  out["shifted_solves"] = diagnostics.shifted_solves;
  out["cg_iterations"] = diagnostics.cg_iterations;
  out["converged"] = diagnostics.converged;
  return out;
}

}

NB_MODULE(_core, m) {

  nb::class_<PyApplyQ>(m, "_ApplyQ")
      .def(nb::init<std::vector<Spin>, std::string, std::vector<Channel>>(),
           nb::arg("two_js"),
           nb::arg("triple_signs"),
           nb::arg("channels"))
      .def("__call__", &apply_array,
           nb::arg("input"))
      .def("apply", &apply_array,
           nb::arg("input"))
      .def("apply_into", &apply_into_array,
           nb::arg("input"),
           nb::arg("output"))
      .def("apply_mpi", &apply_mpi_array,
           nb::arg("input"))
      .def("apply_mpi_into", &apply_mpi_into_array,
           nb::arg("input"),
           nb::arg("output"))
      .def("apply_vector", &apply_vector,
           nb::arg("input"))
      .def("matrix_element",
           [](const PyApplyQ& self, const Channel& bra, const Channel& ket, int i, int j, int k) {
             return self.kernel().matrix_element(bra, ket, i, j, k);
           },
           nb::arg("bra"),
           nb::arg("ket"),
           nb::arg("i"),
           nb::arg("j"),
           nb::arg("k"))
      .def("row_sum_bound",
           [](const PyApplyQ& self) {
             nb::gil_scoped_release release;
             return self.kernel().row_sum_bound();
           })
      .def_prop_ro("dim", [](const PyApplyQ& self) { return self.kernel().dim(); })
      .def_prop_ro("valence", [](const PyApplyQ& self) { return self.kernel().valence(); })
      .def_prop_ro("two_js", [](const PyApplyQ& self) { return self.kernel().two_js(); })
      .def_prop_ro("channels", [](const PyApplyQ& self) { return self.kernel().channels(); })
      .def_prop_ro("sixj_cache_size",
                   [](const PyApplyQ& self) { return self.kernel().sixj_cache_size(); });

  nb::class_<PyVolumeSRQ>(m, "_VolumeSRQ")
      .def(nb::init<std::vector<Spin>, std::string, std::vector<Channel>, double, int, double, int>(),
           nb::arg("two_js"),
           nb::arg("triple_signs"),
           nb::arg("channels"),
           nb::arg("lambda_bound"),
           nb::arg("shifts"),
           nb::arg("cg_tol"),
           nb::arg("cg_maxiter"))
      .def("__call__", &srq_apply_array,
           nb::arg("input"))
      .def("apply", &srq_apply_array,
           nb::arg("input"))
      .def("apply_into", &srq_apply_into_array,
           nb::arg("input"),
           nb::arg("output"))
      .def("apply_mpi", &srq_apply_mpi_array,
           nb::arg("input"))
      .def("apply_mpi_into", &srq_apply_mpi_into_array,
           nb::arg("input"),
           nb::arg("output"))
      .def("diagnostics", &srq_diagnostics_dict)
      .def_prop_ro("dim", [](const PyVolumeSRQ& self) { return self.volume().dim(); })
      .def_prop_ro("valence", [](const PyVolumeSRQ& self) { return self.volume().valence(); })
      .def_prop_ro("two_js", [](const PyVolumeSRQ& self) { return self.volume().two_js(); })
      .def_prop_ro("channels", [](const PyVolumeSRQ& self) { return self.volume().channels(); })
      .def_prop_ro("lambda_bound", [](const PyVolumeSRQ& self) { return self.volume().lambda_bound(); })
      .def_prop_ro("shifts", [](const PyVolumeSRQ& self) { return self.volume().shifts(); })
      .def_prop_ro("cg_tol", [](const PyVolumeSRQ& self) { return self.volume().cg_tol(); })
      .def_prop_ro("cg_maxiter", [](const PyVolumeSRQ& self) { return self.volume().cg_maxiter(); });

  m.def("enumerate_left_comb_channels",
        &btkernel::enumerate_left_comb_channels,
        nb::arg("two_js"));

  m.def("parse_spin_list",
        &btkernel::parse_spin_list,
        nb::arg("text"));

  m.def("parse_triple_signs",
        [](const std::string& text, int valence) {
          nb::list out;
          for (const auto& sign : btkernel::parse_triple_signs(text, valence)) {
            out.append(nb::make_tuple(sign.i, sign.j, sign.k, sign.sign));
          }
          return out;
        },
        nb::arg("text"),
        nb::arg("valence"));

  m.def("openmp_enabled",
        &btkernel::openmp_enabled);
  m.def("openmp_max_threads",
        &btkernel::openmp_max_threads);
  m.def("openmp_set_num_threads",
        &btkernel::openmp_set_num_threads,
        nb::arg("threads"));
  m.def("mpi_enabled",
        &btkernel::mpi_enabled);
  m.def("mpi_initialized",
        &btkernel::mpi_initialized);
  m.def("mpi_finalized",
        &btkernel::mpi_finalized);
  m.def("mpi_init_thread",
        &btkernel::mpi_init_thread);
  m.def("mpi_finalize",
        &btkernel::mpi_finalize);
  m.def("mpi_world_rank",
        &btkernel::mpi_world_rank);
  m.def("mpi_world_size",
        &btkernel::mpi_world_size);

  m.def("_apply_q",
        [](std::vector<Spin> two_js,
           const std::string& triple_signs,
           std::vector<Channel> channels,
           const InputArray& input) {
          PyApplyQ op(std::move(two_js), triple_signs, std::move(channels));
          return apply_array(op, input);
        },
        nb::arg("two_js"),
        nb::arg("triple_signs"),
        nb::arg("channels"),
        nb::arg("input"));
}
