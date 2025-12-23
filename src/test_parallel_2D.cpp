#include "FSI.hpp"

int main(int argc, char *argv[])
{
  // Initialize MPI
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  // Example usage: create FSI object with polynomial degrees
  FSI fsi(2, 1, 2);

  // Run the simulation
  fsi.run();

  return 0;
}