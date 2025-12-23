#include "FSISerial.hpp"

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  // Example usage: create FSI object with mesh file and polynomial degrees
  FSISerial fsi(2, 1, 1);

  // Run the simulation
  fsi.run();

  return 0;
}