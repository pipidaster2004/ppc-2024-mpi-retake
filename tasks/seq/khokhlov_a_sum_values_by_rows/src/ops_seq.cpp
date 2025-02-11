#include "seq/khokhlov_a_sum_values_by_rows/include/ops_sec.hpp"

using namespace std::chrono_literals;

bool khokhlov_a_sum_values_by_rows_seq::Sum_val_by_rows::PreProcessingImpl() {
  // Init vectors
  input_ = std::vector<int>(task_data->inputs_count[0]);
  auto tmp = reinterpret_cast<int*>(task_data->inputs[0]);
  std::copy(tmp, tmp + task_data->inputs_count[0], input_.begin());
  row = task_data->inputs_count[1];
  col = task_data->inputs_count[2];
  // Init value for output
  sum = std::vector<int>(row, 0);
  return true;
}

bool khokhlov_a_sum_values_by_rows_seq::Sum_val_by_rows::ValidationImpl() {
  return (task_data->inputs_count[1] == task_data->outputs_count[0]);
}

bool khokhlov_a_sum_values_by_rows_seq::Sum_val_by_rows::RunImpl() {
  for (int i = 0; i < row; i++) {
    int tmp_sum = 0;
    for (int j = 0; j < col; j++) tmp_sum += input_[i * col + j];
    sum[i] += tmp_sum;
  }
  return true;
}

bool khokhlov_a_sum_values_by_rows_seq::Sum_val_by_rows::PostProcessingImpl() {
  for (int i = 0; i < row; i++) reinterpret_cast<int*>(task_data->outputs[0])[i] = sum[i];
  return true;
}

std::vector<int> khokhlov_a_sum_values_by_rows_seq::getRandomMatrix(int rows, int cols) {
  int sz = rows * cols;
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}