#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace khokhlov_a_multi_integration_monte_karlo_mpi {

class MonteCarloMpi : public ppc::core::Task {
 public:
  explicit MonteCarloMpi(ppc::core::TaskDataPtr task_data, std::function<double(const std::vector<double>&)> f)
   : Task(std::move(task_data)), integrand_(f) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
 boost::mpi::communicator world_;
  size_t dimension_;
  int N_;
  std::vector<double> lower_bound_;
  std::vector<double> upper_bound_;
  std::function<double(const std::vector<double>&)> integrand_;
  double result_;
};

}  // namespace khokhlov_a_multi_integration_monte_karlo_mpi