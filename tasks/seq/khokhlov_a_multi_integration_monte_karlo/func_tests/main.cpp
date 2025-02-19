#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/khokhlov_a_multi_integration_monte_karlo/include/ops_seq.hpp"

TEST(khokhlov_a_multi_integration_monte_karlo_seq, test_invalid_dimension) {
  // create data
  unsigned int dimension = 0;
  std::vector<double> l_bound = {0.0};
  std::vector<double> u_bound = {1.0};
  int n = 100;
  double res = 0.0;

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(dimension);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(l_bound.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(u_bound.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(l_bound.size());
  task_data_seq->inputs_count.emplace_back(u_bound.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));

  // crate task
  khokhlov_a_multi_integration_monte_karlo_seq::MonteCarloSeq monte_carlo(task_data_seq);

  ASSERT_FALSE(monte_carlo.ValidationImpl());
}

TEST(khokhlov_a_multi_integration_monte_karlo_seq, test_invalid_iter) {
  // create data
  unsigned int dimension = 1;
  std::vector<double> l_bound = {0.0};
  std::vector<double> u_bound = {1.0};
  int n = 0;
  double res = 0.0;

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(dimension);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(l_bound.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(u_bound.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(l_bound.size());
  task_data_seq->inputs_count.emplace_back(u_bound.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));

  // crate task
  khokhlov_a_multi_integration_monte_karlo_seq::MonteCarloSeq monte_carlo(task_data_seq);

  ASSERT_FALSE(monte_carlo.ValidationImpl());
}

TEST(khokhlov_a_multi_integration_monte_karlo_seq, test_invalid_bounds) {
  // create data
  unsigned int dimension = 1;
  std::vector<double> l_bound;
  std::vector<double> u_bound;
  int n = 100;
  double res = 0.0;

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(dimension);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(l_bound.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(u_bound.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(l_bound.size());
  task_data_seq->inputs_count.emplace_back(u_bound.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));

  // crate task
  khokhlov_a_multi_integration_monte_karlo_seq::MonteCarloSeq monte_carlo(task_data_seq);

  ASSERT_FALSE(monte_carlo.ValidationImpl());
}

TEST(khokhlov_a_multi_integration_monte_karlo_seq, test_invalid_bounds_1) {
  // create data
  unsigned int dimension = 1;
  std::vector<double> l_bound = {1.0};
  std::vector<double> u_bound = {0.0};
  int n = 100;
  double res = 0.0;

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(dimension);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(l_bound.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(u_bound.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(l_bound.size());
  task_data_seq->inputs_count.emplace_back(u_bound.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));

  // crate task
  khokhlov_a_multi_integration_monte_karlo_seq::MonteCarloSeq monte_carlo(task_data_seq);

  ASSERT_FALSE(monte_carlo.ValidationImpl());
}

TEST(khokhlov_a_multi_integration_monte_karlo_seq, test_invalid_bounds_2) {
  // create data
  unsigned int dimension = 2;
  std::vector<double> l_bound = {0.0, 0.0};
  std::vector<double> u_bound = {1.0};
  int n = 100;
  double res = 0.0;

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(dimension);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(l_bound.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(u_bound.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(l_bound.size());
  task_data_seq->inputs_count.emplace_back(u_bound.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));

  // crate task
  khokhlov_a_multi_integration_monte_karlo_seq::MonteCarloSeq monte_carlo(task_data_seq);

  ASSERT_FALSE(monte_carlo.ValidationImpl());
}

TEST(khokhlov_a_multi_integration_monte_karlo_seq, test_1_dim) {
  // create data
  unsigned int dimension = 1;
  std::vector<double> l_bound = {0.0};
  std::vector<double> u_bound = {1.0};
  int n = 1000;
  double res = 0.0;

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(dimension);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(l_bound.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(u_bound.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(l_bound.size());
  task_data_seq->inputs_count.emplace_back(u_bound.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));

  // crate task
  khokhlov_a_multi_integration_monte_karlo_seq::MonteCarloSeq monte_carlo(task_data_seq);
  monte_carlo.integrand = [](const std::vector<double> &point) { return exp(point[0]); };
  ASSERT_TRUE(monte_carlo.ValidationImpl());
  monte_carlo.PreProcessingImpl();
  monte_carlo.RunImpl();
  monte_carlo.PostProcessingImpl();
  double expected = 1.718;
  ASSERT_NEAR(res, expected, 1e-1);
}

TEST(khokhlov_a_multi_integration_monte_karlo_seq, test_2_dim) {
  // create data
  unsigned int dimension = 2;
  std::vector<double> l_bound = {0.0, 0.0};
  std::vector<double> u_bound = {1.0, 1.0};
  int n = 1000;
  double res = 0.0;

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(dimension);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(l_bound.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(u_bound.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(l_bound.size());
  task_data_seq->inputs_count.emplace_back(u_bound.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));

  // crate task
  khokhlov_a_multi_integration_monte_karlo_seq::MonteCarloSeq monte_carlo(task_data_seq);
  monte_carlo.integrand = [](const std::vector<double> &point) { return point[0] + point[1]; };
  ASSERT_TRUE(monte_carlo.ValidationImpl());
  monte_carlo.PreProcessingImpl();
  monte_carlo.RunImpl();
  monte_carlo.PostProcessingImpl();

  double exp = 1.0;
  ASSERT_NEAR(res, exp, 1e-1);
}

TEST(khokhlov_a_multi_integration_monte_karlo_seq, test_2_dim_cos) {
  // create data
  const int dimension = 2;
  std::vector<double> l_bound = {0.0, 0.0};
  std::vector<double> u_bound = {std::numbers::pi / 2.0, std::numbers::pi / 2.0};
  int n = 1000;
  double res = 0.0;

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(dimension);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(l_bound.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(u_bound.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(l_bound.size());
  task_data_seq->inputs_count.emplace_back(u_bound.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));

  // crate task
  khokhlov_a_multi_integration_monte_karlo_seq::MonteCarloSeq monte_carlo(task_data_seq);
  monte_carlo.integrand = [](const std::vector<double> &point) { return cos(point[0]) * cos(point[1]); };
  ASSERT_TRUE(monte_carlo.ValidationImpl());
  monte_carlo.PreProcessingImpl();
  monte_carlo.RunImpl();
  monte_carlo.PostProcessingImpl();

  double exp = 1.0;
  ASSERT_NEAR(res, exp, 1e-1);
}

TEST(khokhlov_a_multi_integration_monte_karlo_seq, test_3_dim) {
  // create data
  const int dimension = 3;
  std::vector<double> l_bound = {0.0, 0.0, 0.0};
  std::vector<double> u_bound = {1.0, 1.0, 1.0};
  int n = 1000;
  double res = 0.0;

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(dimension);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(l_bound.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(u_bound.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(l_bound.size());
  task_data_seq->inputs_count.emplace_back(u_bound.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));

  // crate task
  khokhlov_a_multi_integration_monte_karlo_seq::MonteCarloSeq monte_carlo(task_data_seq);
  monte_carlo.integrand = [](const std::vector<double> &point) { return point[0] + point[1] + point[2]; };
  ASSERT_TRUE(monte_carlo.ValidationImpl());
  monte_carlo.PreProcessingImpl();
  monte_carlo.RunImpl();
  monte_carlo.PostProcessingImpl();

  double exp = 1.5;
  ASSERT_NEAR(res, exp, 1e-1);
}

TEST(khokhlov_a_multi_integration_monte_karlo_seq, test_3_dim_1) {
  // create data
  const int dimension = 3;
  std::vector<double> l_bound = {0.0, 0.0, 0.0};
  std::vector<double> u_bound = {1.0, 1.0, 1.0};
  int n = 1000;
  double res = 0.0;

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(dimension);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(l_bound.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(u_bound.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(l_bound.size());
  task_data_seq->inputs_count.emplace_back(u_bound.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));

  // crate task
  khokhlov_a_multi_integration_monte_karlo_seq::MonteCarloSeq monte_carlo(task_data_seq);
  monte_carlo.integrand = [](const std::vector<double> &point) { return point[0] * point[1] * point[2]; };
  ASSERT_TRUE(monte_carlo.ValidationImpl());
  monte_carlo.PreProcessingImpl();
  monte_carlo.RunImpl();
  monte_carlo.PostProcessingImpl();

  double exp = 0.125;
  ASSERT_NEAR(res, exp, 1e-1);
}

TEST(khokhlov_a_multi_integration_monte_karlo_seq, test_4_dim) {
  // create data
  const int dimension = 4;
  std::vector<double> l_bound = {0.0, 0.0, 0.0, 0.0};
  std::vector<double> u_bound = {1.0, 1.0, 1.0, 1.0};
  int n = 1000;
  double res = 0.0;

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(dimension);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(l_bound.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(u_bound.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->inputs_count.emplace_back(l_bound.size());
  task_data_seq->inputs_count.emplace_back(u_bound.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));

  // crate task
  khokhlov_a_multi_integration_monte_karlo_seq::MonteCarloSeq monte_carlo(task_data_seq);
  monte_carlo.integrand = [](const std::vector<double> &point) {
    return (point[0] * point[1]) + (point[2] * point[3]);
  };
  ASSERT_TRUE(monte_carlo.ValidationImpl());
  monte_carlo.PreProcessingImpl();
  monte_carlo.RunImpl();
  monte_carlo.PostProcessingImpl();

  double exp = 0.5;
  ASSERT_NEAR(res, exp, 1e-1);
}