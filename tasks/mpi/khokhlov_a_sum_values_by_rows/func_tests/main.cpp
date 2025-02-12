#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include "mpi/khokhlov_a_sum_values_by_rows/include/ops_mpi.hpp"

TEST(khokhlov_a_sum_values_by_rows_mpi, test_empty_matrix) {
  boost::mpi::communicator world;

  int cols = 0;
  int rows = 0;

  // Create data
  std::vector<int> in = {};
  std::vector<int> out_par = {};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_par->inputs_count.emplace_back(in.size());
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_par->outputs_count.emplace_back(out_par.size());
  }

  khokhlov_a_sum_values_by_rows_mpi::SumValByRowsMpi SumValByRowsMpi(task_data_par);
  ASSERT_EQ(SumValByRowsMpi.ValidationImpl(), true);
  SumValByRowsMpi.PreProcessingImpl();
  SumValByRowsMpi.RunImpl();
  SumValByRowsMpi.PostProcessingImpl();
}

TEST(khokhlov_a_sum_values_by_rows_mpi, test_const_matrix) {
  boost::mpi::communicator world;

  int cols = 50;
  int rows = 100;

  // Create data
  std::vector<int> in(cols * rows, 0);
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) in[i] += (i * cols + j);
  std::vector<int> out_par(rows, 0);

  std::vector<int> expect(rows, 0);
  for (int i = 0; i < rows; i++) {
    int tmp_sum = 0;
    for (int j = 0; j < cols; j++) {
      tmp_sum += in[i * cols + j];
    }
    expect[i] += tmp_sum;
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_par->inputs_count.emplace_back(in.size());
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_par->outputs_count.emplace_back(out_par.size());
  }

  khokhlov_a_sum_values_by_rows_mpi::SumValByRowsMpi SumValByRowsMpi(task_data_par);
  ASSERT_EQ(SumValByRowsMpi.ValidationImpl(), true);
  SumValByRowsMpi.PreProcessingImpl();
  SumValByRowsMpi.RunImpl();
  SumValByRowsMpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(out_par, expect);
  }
}

TEST(khokhlov_a_sum_values_by_rows_mpi, test_const_diag_matrix_with_negativ) {
  boost::mpi::communicator world;

  int cols = 50;
  int rows = 100;

  // Create data
  std::vector<int> in(cols * rows, 0);
  for (int i = 0; i < rows; i++)
    for (int j = i; j < cols; j++) in[i] += -(i * cols + j);
  std::vector<int> out_par(rows, 0);

  std::vector<int> expect(rows, 0);
  for (int i = 0; i < rows; i++) {
    int tmp_sum = 0;
    for (int j = 0; j < cols; j++) {
      tmp_sum += in[i * cols + j];
    }
    expect[i] += tmp_sum;
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_par->inputs_count.emplace_back(in.size());
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_par->outputs_count.emplace_back(out_par.size());
  }

  khokhlov_a_sum_values_by_rows_mpi::SumValByRowsMpi SumValByRowsMpi(task_data_par);
  ASSERT_EQ(SumValByRowsMpi.ValidationImpl(), true);
  SumValByRowsMpi.PreProcessingImpl();
  SumValByRowsMpi.RunImpl();
  SumValByRowsMpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(out_par, expect);
  }
}

TEST(khokhlov_a_sum_values_by_rows_mpi, test_random_matrix) {
  boost::mpi::communicator world;
  int cols = 20;
  int rows = 13;

  // Create data
  std::vector<int> in = khokhlov_a_sum_values_by_rows_mpi::GetRandomMatrix(rows * cols);

  std::vector<int> out_par(rows, 0);

  std::vector<int> exp(rows, 0);
  for (int i = 0; i < rows; i++) {
    int tmp_sum = 0;
    for (int j = 0; j < cols; j++) {
      tmp_sum += in[i * cols + j];
    }
    exp[i] += tmp_sum;
  }
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_par = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_par->inputs_count.emplace_back(in.size());
    task_data_par->inputs_count.emplace_back(rows);
    task_data_par->inputs_count.emplace_back(cols);
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_par.data()));
    task_data_par->outputs_count.emplace_back(out_par.size());
  }

  khokhlov_a_sum_values_by_rows_mpi::SumValByRowsMpi SumValByRowsMpi(task_data_par);
  ASSERT_EQ(SumValByRowsMpi.ValidationImpl(), true);
  SumValByRowsMpi.PreProcessingImpl();
  SumValByRowsMpi.RunImpl();
  SumValByRowsMpi.PostProcessingImpl();

  if (world.rank() == 0) {
    ASSERT_EQ(out_par, exp);
  }
}