#include "mpi/khokhlov_a_multi_integration_monte_karlo/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>

bool khokhlov_a_multi_integration_monte_karlo_mpi::MonteCarloMpi::PreProcessingImpl() { return true; }

bool khokhlov_a_multi_integration_monte_karlo_mpi::MonteCarloMpi::ValidationImpl() { return true; }

bool khokhlov_a_multi_integration_monte_karlo_mpi::MonteCarloMpi::RunImpl() { return true; }

bool khokhlov_a_multi_integration_monte_karlo_mpi::MonteCarloMpi::PostProcessingImpl() { return true; }