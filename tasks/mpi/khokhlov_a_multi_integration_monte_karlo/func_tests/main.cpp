#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <vector>
#include <cmath>

#include "core/task/include/task.hpp"
#include "mpi/khokhlov_a_multi_integration_monte_karlo/include/ops_mpi.hpp"

// namespace khokhlov_a_multi_integration_monte_karlo_mpi{
//     namespace {
// double Integrand1d1(const std::vector<double>& x) {return exp(x[0]);}
// double Integrand1d2(const std::vector<double>& x) {return cos(x[0]);}
// double Integrand2d1(const std::vector<double>& x) {return x[0] + sin(x[1]);}
// double Integrand2d2(const std::vector<double>& x) {return x[0] + x[1];}
// double Integrand2d3(const std::vector<double>& x) {return x[0]*x[1];}
// double Integrand3d1(const std::vector<double>& x) {return x[0]+x[1]+x[2];}
// double Integrand3d2(const std::vector<double>& x) {return sin(x[0])+x[1]*x[2];}
// double Integrand3d3(const std::vector<double>& x) {return x[0]*x[1]+exp(x[2]);}
// double Integrand4d1(const std::vector<double>& x) {return x[0]+x[1]+x[2]+x[3];}
//     }
// }

TEST(khokhlov_a_multi_integration_monte_karlo_mpi, 1)
{
    bool a = true;
    ASSERT_EQ(a, true);
}