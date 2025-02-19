#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/khokhlov_a_multi_integration_monte_karlo/include/ops_mpi.hpp"

TEST(khokhlov_a_multi_integration_monte_karlo_mpi, test_pipline_run_mpi) {
  boost::mpi::communicator world;
  const int dimension = 3;
  std::vector<double> l_bound = {0.0, 0.0, 0.0};
  std::vector<double> u_bound = {1.0, 1.0, 1.0};
  int n = 5000000;
  double res = 0.0;

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs_count.emplace_back(dimension);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(l_bound.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(u_bound.data()));
    task_data_mpi->inputs_count.emplace_back(n);
    task_data_mpi->inputs_count.emplace_back(l_bound.size());
    task_data_mpi->inputs_count.emplace_back(u_bound.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));
  }

  // crate task
  auto test_task_mpi = std::make_shared<khokhlov_a_multi_integration_monte_karlo_mpi::MonteCarloMpi>(task_data_mpi);
  test_task_mpi->integrand = [](const std::vector<double> &point) {
    return cos(point[0]) + (sin(point[1]) * cos(point[2]));
  };

  // create perf attrib
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    double expected = 1.19551;
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_NEAR(res, expected, 1e-1);
  }
}

TEST(khokhlov_a_multi_integration_monte_karlo_mpi, test_task_run_mpi) {
  boost::mpi::communicator world;
  const int dimension = 3;
  std::vector<double> l_bound = {0.0, 0.0, 0.0};
  std::vector<double> u_bound = {1.0, 1.0, 1.0};
  int n = 5000000;
  double res = 0.0;

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs_count.emplace_back(dimension);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(l_bound.data()));
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(u_bound.data()));
    task_data_mpi->inputs_count.emplace_back(n);
    task_data_mpi->inputs_count.emplace_back(l_bound.size());
    task_data_mpi->inputs_count.emplace_back(u_bound.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(&res));
  }

  // crate task
  auto test_task_mpi = std::make_shared<khokhlov_a_multi_integration_monte_karlo_mpi::MonteCarloMpi>(task_data_mpi);
  test_task_mpi->integrand = [](const std::vector<double> &point) {
    return cos(point[0]) + (sin(point[1]) * cos(point[2]));
  };

  // create perf attrib
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    double expected = 1.19551;
    ASSERT_NEAR(res, expected, 1e-1);
  }
}