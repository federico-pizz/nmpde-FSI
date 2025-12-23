// Base
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

// Distributed
#include <deal.II/distributed/fully_distributed_tria.h>

// DoFs
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

// Finite Elements
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

// Grid
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>

// hp
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

// Linear Algebra
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/vector.h>

// Numerics
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

// Standard library
#include <fstream>
#include <iostream>
#include <filesystem>
#include <memory>

using namespace dealii;

class FSI {
public:
  // Preconditioner for parallel
  class FSIPreconditioner {
  public:
    void initialize(const TrilinosWrappers::BlockSparseMatrix &velocity_stiffness_,
                    const TrilinosWrappers::BlockSparseMatrix &pressure_mass_,
                    const TrilinosWrappers::BlockSparseMatrix &displacement_stiffness_,
                    const TrilinosWrappers::BlockSparseMatrix &B10_,
                    const TrilinosWrappers::BlockSparseMatrix &B20_,
                    const TrilinosWrappers::BlockSparseMatrix &B21_,
                    const std::vector<std::vector<bool>> &displacement_constant_modes) {
      velocity_stiffness = &velocity_stiffness_;
      pressure_mass      = &pressure_mass_;
      B10                = &B10_;
      B20                = &B20_;
      B21                = &B21_;
      displacement_stiffness = &displacement_stiffness_;

      // Trilinos AMG preconditioners
      TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
      amg_data.constant_modes = displacement_constant_modes;
      amg_data.elliptic = true;
      amg_data.higher_order_elements = false;
      amg_data.smoother_sweeps = 3;
      amg_data.w_cycle = false;
      amg_data.aggregation_threshold = 1e-3;
      // Use the displacement block (2,2) for AMG initialization
      preconditioner_displacement.initialize(displacement_stiffness_.block(2, 2), amg_data);

      TrilinosWrappers::PreconditionAMG::AdditionalData amg_data_vel;
      amg_data_vel.elliptic = true;
      amg_data_vel.higher_order_elements = true;
      amg_data_vel.smoother_sweeps = 2;
      amg_data_vel.aggregation_threshold = 1e-3;
      // Use the velocity block (0,0) for AMG initialization
      preconditioner_velocity.initialize(velocity_stiffness_.block(0, 0), amg_data_vel);
      // Temporary vectors are reinitialized lazily in vmult() to match partitioning
    }

    void vmult(TrilinosWrappers::MPI::BlockVector &dst,
               const TrilinosWrappers::MPI::BlockVector &src) const {

      // Ensure temporaries have correct partitioning
      if (tmp_p.size() != src.block(1).size())
        tmp_p.reinit(src.block(1), /*fast=*/false, /*reset=*/false);
      if (tmp_d.size() != src.block(2).size())
        tmp_d.reinit(src.block(2), /*fast=*/false, /*reset=*/false);
      if (intermediate_tmp.size() != src.block(2).size())
        intermediate_tmp.reinit(src.block(2), /*fast=*/false, /*reset=*/false);

      // Fluid block
      SolverControl solver_control_vel(2000, 1e-2 * src.block(0).l2_norm());
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_vel(solver_control_vel);
      dst.block(0) = 0;
      solver_gmres_vel.solve(*velocity_stiffness, dst.block(0), src.block(0), preconditioner_velocity);

      // Pressure block
      B10->vmult(tmp_p, dst.block(0));
      tmp_p.sadd(-1.0, 1.0, src.block(1));
      SolverControl solver_control_pres(2000, 1e-2 * tmp_p.l2_norm());
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver_cg_pres(solver_control_pres);
      dst.block(1) = 0;
      solver_cg_pres.solve(*pressure_mass, dst.block(1), tmp_p, preconditioner_velocity);
      
      // Displacement block
      B20->vmult(tmp_d, dst.block(0));
      B21->vmult(intermediate_tmp, dst.block(1));
      tmp_d.add(1.0, intermediate_tmp);
      tmp_d.sadd(-1.0, 1.0, src.block(2));
      SolverControl solver_control_disp(2000, 1e-2 * tmp_d.l2_norm());
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_disp(solver_control_disp);
      dst.block(2) = 0;
      solver_gmres_disp.solve(*displacement_stiffness, dst.block(2), tmp_d, preconditioner_displacement);
    }
  protected:
    const TrilinosWrappers::BlockSparseMatrix *velocity_stiffness;
    const TrilinosWrappers::BlockSparseMatrix *displacement_stiffness;
    const TrilinosWrappers::BlockSparseMatrix *B10;
    const TrilinosWrappers::BlockSparseMatrix *B20;
    const TrilinosWrappers::BlockSparseMatrix *B21;
    TrilinosWrappers::PreconditionAMG preconditioner_velocity;
    TrilinosWrappers::PreconditionAMG preconditioner_displacement;
    const TrilinosWrappers::BlockSparseMatrix *pressure_mass;
    mutable TrilinosWrappers::MPI::Vector tmp_p;
    mutable TrilinosWrappers::MPI::Vector tmp_d;
    mutable TrilinosWrappers::MPI::Vector intermediate_tmp;
  };

  FSIPreconditioner preconditioner;
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 2;  
  
  class InletVelocity : public Function<dim> {
  public:
    InletVelocity() : Function<dim>(dim + 1 + dim) {}

    virtual double value(const Point<dim> &p, const unsigned int component) const override {
      Assert(component < this->n_components, ExcIndexRange(component, 0, this->n_components));
      if (component == dim - 1) {
        switch (dim) {
          case 2:
            return std::sin(numbers::PI * p[0]);
          case 3:
            return std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
          default:
            Assert(false, ExcNotImplemented());
        }
      }
      return 0;
    }

    virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override {
      for (unsigned int c = 0; c < this->n_components; ++c)
        values(c) = this->value(p, c);
    }
  };


  // Constructor
  FSI(const std::string &mesh_file_name_,
      const unsigned int &degree_velocity_,
      const unsigned int &degree_pressure_,
      const unsigned int &degree_displacement_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
      mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
      pcout(std::cout, mpi_rank == 0),
      mesh_file_name(mesh_file_name_),
      degree_velocity(degree_velocity_),
      degree_pressure(degree_pressure_),
      degree_displacement(degree_displacement_),
      mesh(MPI_COMM_WORLD) {}

  // Convenience constructor when using the procedurally generated grid
  FSI(const unsigned int &degree_velocity_,
      const unsigned int &degree_pressure_,
      const unsigned int &degree_displacement_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
      mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
      pcout(std::cout, mpi_rank == 0),
      mesh_file_name(""),
      degree_velocity(degree_velocity_),
      degree_pressure(degree_pressure_),
      degree_displacement(degree_displacement_),
      mesh(MPI_COMM_WORLD) {}

  // Setup system (mesh, FE space, DoF handler, and linear system).
  void make_grid();
  void setup();
  void assemble_system();
  void assemble_interface_term(
    const FEFaceValuesBase<dim> &elasticity_fe_face_values,
    const FEFaceValuesBase<dim> &stokes_fe_face_values,
    std::vector<Tensor<1, dim>> &elasticity_phi,
    std::vector<Tensor<2, dim>> &stokes_grad_phi_u,
    std::vector<double> &stokes_phi_p,
    FullMatrix<double> &local_interface_matrix) const;
  void solve();
  void output(const unsigned int refinement_cycle);
  void refine_mesh();
  void run();

protected:
  // Domain identifiers for material ID tagging.
  static constexpr types::material_id fluid_domain_id = 0;
  static constexpr types::material_id solid_domain_id = 1;

  // MPI information. ////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Physical and material parameters. ////////////////////////////////////////

  // Kinematic viscosity [m2/s]. - deal-ii uses 2 for some reason
  const double nu = 2;

  // Outlet pressure [Pa].
  const double p_out = 10;

  // Lamé parameters.
  const double mu     = 1.0;
  const double lambda = 10.0;

  // Forcing term.
  Tensor<1, dim> f;

  // Dirichlet datum. Can be omitted in our problem
  // FunctionG function_g; 

  // Inlet velocity.
  InletVelocity inlet_velocity;

  // Discretization parameters and objects. ///////////////////////////////////

  // Mesh file name.
  const std::string mesh_file_name;

  // Polynomial degree used for velocity.
  const unsigned int degree_velocity;

  // Polynomial degree used for pressure.
  const unsigned int degree_pressure;

  // Polynomial degree used for displacement
  const unsigned int degree_displacement;

  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // DoF handler and FE collections
  DoFHandler<dim> dof_handler;
  hp::FECollection<dim> fe_collection;
  std::unique_ptr<FESystem<dim>> stokes_fe;
  std::unique_ptr<FESystem<dim>> elasticity_fe;
  std::unique_ptr<QGauss<dim>> stokes_quadrature;
  std::unique_ptr<QGauss<dim>> elasticity_quadrature;
  std::unique_ptr<QGauss<dim - 1>> common_face_quadrature;

  // Constraints and index sets
  AffineConstraints<double> constraints;
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;
  std::vector<IndexSet> block_owned_dofs;
  std::vector<IndexSet> block_relevant_dofs;

  // System matrix (FSI operator)
  TrilinosWrappers::BlockSparseMatrix system_matrix;

  // Pressure mass matrix, needed for preconditioning.
  TrilinosWrappers::BlockSparseMatrix pressure_mass;

  // Mass matrix (velocity + displacement only)
  TrilinosWrappers::BlockSparseMatrix mass_matrix;

  // Right-hand side vector in the linear system.
  TrilinosWrappers::MPI::BlockVector system_rhs;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::BlockVector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::BlockVector solution;

  // Helper functions. ///////////////////////////////////////////////////////////

  static bool cell_is_in_fluid_domain(
    const typename DoFHandler<dim>::cell_iterator &cell);

  static bool cell_is_in_solid_domain(
    const typename DoFHandler<dim>::cell_iterator &cell);


private:
  void assemble_mass_matrix();

};