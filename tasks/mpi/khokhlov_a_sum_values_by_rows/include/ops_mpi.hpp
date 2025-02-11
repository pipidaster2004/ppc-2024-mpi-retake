#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

namespace khokhlov_a_sum_values_by_rows_mpi {

std::vector<int> getRandomMatrix(int size);

class Sum_val_by_rows_mpi : public ppc::core::Task {
 public:
  explicit Sum_val_by_rows_mpi(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, local_input_;
  int row, col;
  std::vector<int> sum;
  boost::mpi::communicator world;
};

}  // namespace khokhlov_a_sum_values_by_rows_mpi