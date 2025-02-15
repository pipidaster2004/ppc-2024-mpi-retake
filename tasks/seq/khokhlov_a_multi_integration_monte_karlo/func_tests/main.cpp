#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "seq/khokhlov_a_multi_integration_monte_karlo/include/ops_seq.hpp"

double Integrand1d1(const std::vector<double>& x) {return exp(x[0]);}
double Integrand1d2(const std::vector<double>& x) {return cos(x[0]);}
double Integrand2d1(const std::vector<double>& x) {return x[0] + sin(x[1]);}
double Integrand2d2(const std::vector<double>& x) {return x[0] + x[1];}
double Integrand2d3(const std::vector<double>& x) {return x[0]*x[1];}
double Integrand3d1(const std::vector<double>& x) {return x[0]+x[1]+x[2];}
double Integrand3d2(const std::vector<double>& x) {return sin(x[0])+x[1]*x[2];}
double Integrand3d3(const std::vector<double>& x) {return x[0]*x[1]+exp(x[2]);}
double Integrand4d1(const std::vector<double>& x) {return x[0]+x[1]+x[2]+x[3];}

TEST(khokhlov_a_multi_integration_monte_karlo_seq, test_empty_bounds)
{
    // create data
    const int dimension = 1;
    std::vector<int> l_bound;
    std::vector<int> u_bound;
    int n = 100;
    double res = 0.0;
  
    // create task data
    auto task_data_seq = std::make_shared<ppc::core::TaskData>();
    task_data_seq->inputs_count.emplace_back(dimension);
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(l_bound.data()));
    task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(u_bound.data()));
    task_data_seq->inputs_count.emplace_back(n);
    task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&res));
  
    // crate task
    khokhlov_a_multi_integration_monte_karlo_seq::MonteCarloSeq monte_carlo(task_data_seq, Integrand1d1);
    ASSERT_FALSE(monte_carlo.ValidationImpl());
}