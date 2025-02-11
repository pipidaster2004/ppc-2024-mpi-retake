// Copyright 2023 Nesterov Alexander
#pragma once

#include <random>
#include <vector>

#include "core/task/include/task.hpp"

namespace khokhlov_a_sum_values_by_rows_seq {

std::vector<int> getRandomMatrix(int rows, int cols);

class Sum_val_by_rows : public ppc::core::Task {
 public:
  explicit Sum_val_by_rows(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  int row, col;
  std::vector<int> sum;
};

}  // namespace khokhlov_a_sum_values_by_rows_seq