// Copyright 2023 Nesterov Alexander
#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace khokhlov_a_sum_values_by_rows_seq {

std::vector<int> GetRandomMatrix(int rows, int cols);

class SumValByRows : public ppc::core::Task {
 public:
  explicit SumValByRows(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  int row_, col_;
  std::vector<int> sum_;
};

}  // namespace khokhlov_a_sum_values_by_rows_seq