#include <gtest/gtest.h>

#include <cmath>
#include <vector>
// #include <random>

#include "seq/khokhlov_a_multi_integration_monte_karlo/include/ops_seq.hpp"

namespace khokhlov_a_multi_integration_monte_carlo_seq {
namespace {
double CalcEtalon(std::vector<double> &l_bound, std::vector<double> &u_bound) {
  double ans = 1;
  for (size_t i = 0; i < l_bound.size(); i++) {
    ans *= u_bound[i] - l_bound[i];
  }
  return ans;
}
}  // namespace
}  // namespace khokhlov_a_multi_integration_monte_carlo_seq

TEST(khokhlov_a_multi_integration_monte_karlo_seq, test_empty_bounds) {
  // create data
  const int dimension = 1;
  std::vector<double> l_bound = {0.0};
  std::vector<double> u_bound = {1.0};
  int n = 100;
  double res = 0.0;
  double etalon = khokhlov_a_multi_integration_monte_carlo_seq::CalcEtalon(l_bound, u_bound);

  // create task data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs_count.emplace_back(dimension);
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(l_bound.data()));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(u_bound.data()));
  task_data_seq->inputs_count.emplace_back(n);
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));

  // crate task
  khokhlov_a_multi_integration_monte_karlo_seq::MonteCarloSeq monte_carlo(task_data_seq);
  monte_carlo.integrand = [](const std::vector<double> &point) { return exp(point[0]); };
  ASSERT_TRUE(monte_carlo.ValidationImpl());
  monte_carlo.PreProcessingImpl();
  monte_carlo.RunImpl();
  monte_carlo.PostProcessingImpl();
  ASSERT_LE(res, etalon);
}